"""Frozen multilingual E5 token interaction for scoring a small candidate pool."""

from pathlib import Path
import os
import re
import tempfile

from filelock import FileLock
import numpy as np

from mailsearch import E5_QUERY_GRAPH
from mailsearch.dense import DIMENSION, _index_name, semantic_text
from mailsearch.store import CACHE

TOKEN_LIMIT = 72
PROJECTED_DIM = 96
SEED = 271828
COORDINATES = np.sort(
    np.random.default_rng(SEED).choice(DIMENSION, PROJECTED_DIM, replace=False)
)
CONTENT = re.compile(r"[^\W_]", re.UNICODE)


FEATURE_NAMES = ["mean_maxsim", "q25_maxsim", "best_maxsim", "pooled_cosine", "residual_maxsim"]


def unit_vectors(values):
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-5)


class TokenIndex:
    """Quantized per-token vectors. A zero vector marks a masked token position."""

    def __init__(self, docs, dense, cache_dir=CACHE):
        self.dense = dense
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        graph = Path(E5_QUERY_GRAPH).stem
        name = f"late-v1-{graph}-{TOKEN_LIMIT}-{PROJECTED_DIM}-{SEED}-{_index_name(docs)}"
        path = cache_dir / name
        shape = (len(docs), TOKEN_LIMIT, PROJECTED_DIM)
        with FileLock(str(path) + ".lock"):
            if not path.exists():
                self._build(docs, path, shape)
            self.vectors = np.load(path, mmap_mode="r", allow_pickle=False)
            if self.vectors.shape != shape or self.vectors.dtype != np.int8:
                raise ValueError(f"Invalid E5 token index at {path}")

    def _encode(self, texts, prefix):
        inputs = [prefix + text for text in texts]
        with self.dense._tokenizer_lock:
            batch = self.dense.tokenizer(
                inputs, padding=True, truncation=True, max_length=TOKEN_LIMIT,
                return_token_type_ids=True, return_offsets_mapping=True, return_tensors="np",
            )
        offsets = batch.pop("offset_mapping")
        keep = np.zeros(offsets.shape[:2], dtype=bool)
        for i, (text, row) in enumerate(zip(inputs, offsets, strict=True)):
            for j, (start, end) in enumerate(row):
                # The envelope prefix, special tokens, punctuation and padding do not vote.
                if end > len(prefix) and CONTENT.search(text, max(start, len(prefix)), end):
                    keep[i, j] = True
        hidden = self.dense.query_model.run(["last_hidden_state"], dict(batch))[0]
        selected = hidden[:, :, COORDINATES] * keep[:, :, None]
        return selected

    def _build(self, docs, path, shape):
        # Length bucketing limits padding in the local quantized CPU encoder.
        texts = [semantic_text(doc["doc"][:2600]) for doc in docs]
        order = sorted(range(len(docs)), key=lambda i: len(texts[i]))
        fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix=".tmp")
        os.close(fd)
        try:
            out = np.lib.format.open_memmap(temp, mode="w+", dtype=np.int8, shape=shape)
            print(f"Indexing {len(order)} E5 token documents on CPU", flush=True)
            for start in range(0, len(order), 16):
                rows = order[start:start + 16]
                tokens = self._encode([texts[i] for i in rows], "passage: ")
                scale = np.maximum(1e-8, np.max(np.abs(tokens), axis=2, keepdims=True))
                quantized = np.rint(127 * tokens / scale).astype(np.int8)
                out[rows, :quantized.shape[1], :] = quantized
                if start and start % 1024 == 0:
                    print(f"E5 token index: {start}/{len(order)}", flush=True)
            out.flush()
            del out
            os.replace(temp, path)
        finally:
            Path(temp).unlink(missing_ok=True)

    @staticmethod
    def _align(documents, tokens):
        # One matrix multiplication for all tokens in this block of candidates.
        sim = (documents.reshape(-1, PROJECTED_DIM) @ tokens.T).reshape(
            len(documents), TOKEN_LIMIT, len(tokens)
        )
        available = np.any(documents, axis=2, keepdims=True)
        sim = np.where(available, sim, -1.0)
        return np.where(available.any(axis=1), sim.max(axis=1), 0)

    def features(self, query, indices):
        """Token alignment scores for zero-based document row indices."""
        rows = np.asarray(indices, dtype=np.intp)
        result = np.zeros((len(rows), len(FEATURE_NAMES)), dtype=np.float32)
        if not len(rows):
            return result
        tokens = self._encode([query], "query: ")[0]
        tokens = unit_vectors(tokens[np.any(tokens, axis=1)])
        if not len(tokens):
            return result
        q_center = unit_vectors(tokens.sum(axis=0))
        q_residual = unit_vectors(tokens - (tokens @ q_center)[:, None] * q_center)
        q_residual = q_residual[np.any(q_residual, axis=1)]
        for start in range(0, len(rows), 128):
            block = rows[start:start + 128]
            documents = unit_vectors(self.vectors[block].astype(np.float32))
            maxima = self._align(documents, tokens)
            features = result[start:start + len(block)]
            features[:, 0] = maxima.mean(axis=1)
            features[:, 1] = np.quantile(maxima, 0.25, axis=1)
            features[:, 2] = maxima.max(axis=1)
            d_center = unit_vectors(documents.sum(axis=1))
            features[:, 3] = d_center @ q_center
            if len(q_residual):
                center = d_center[:, None, :]
                dot = (documents * center).sum(axis=2, keepdims=True)
                d_residual = unit_vectors(documents - dot * center)
                features[:, 4] = self._align(d_residual, q_residual).mean(axis=1)
        return result
