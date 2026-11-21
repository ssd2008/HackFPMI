"""Local multilingual E5 document and query vectors with a reusable on-disk index."""

from __future__ import annotations

from copy import copy
import hashlib
import os
from pathlib import Path
import re
import tempfile
import threading

import numpy as np
import torch
from filelock import FileLock
from transformers import AutoModel, AutoTokenizer

from . import E5_MODEL as MODEL, E5_REVISION as REVISION
from .private_cache import protect_tokenizer

DIMENSION = 384
TOKEN_LIMIT = 120
# Identifiers remain in the separate lexical index. Long opaque values push the
# actual message out of a short encoder window and add little semantic signal.
_NOISE = re.compile(
    r"https?://[^\s}<>]+|[\w.+-]+@[\w.-]+|syn-[a-f0-9]+|\d+(?:[.:,-]\d+)*|"
    r"(?im:^\s*(?:текст письма|тема|от)\s*:\s*)",
    re.I,
)


def semantic_text(text: str) -> str:
    return " ".join(_NOISE.sub(" ", text).split())


def _index_name(documents: list[dict]) -> str:
    digest = hashlib.sha256(f"v2|{MODEL}|{REVISION}|{TOKEN_LIMIT}".encode())
    for doc in documents:
        digest.update(doc["doc_id"].encode())
        digest.update(b"\0")
        digest.update(doc["doc"].encode())
        digest.update(b"\0")
    return f"e5-{digest.hexdigest()[:24]}.npy"


class DenseIndex:
    def __init__(self, documents: list[dict], cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / _index_name(documents)
        self._query_lock = threading.Lock()
        torch.set_num_threads(min(4, torch.get_num_threads()))
        # The pinned revision can be used entirely offline after first download.
        options = {"revision": REVISION, "local_files_only": True}
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL, **options)
            self.model = AutoModel.from_pretrained(MODEL, **options)
        except OSError as exc:
            raise RuntimeError(
                f"Отсутствуют локальные веса {MODEL}. Запустите python -m mailsearch.prepare."
            ) from exc
        self.model.eval()
        with FileLock(str(path) + ".lock"):
            if not path.exists():
                self._build(documents, path)
                # Building can cast parameters to fp16 on the GPU. Always rank
                # queries with the same original fp32 weights across restarts.
                self.model = AutoModel.from_pretrained(MODEL, **options).eval()
            self.vectors = np.load(path, mmap_mode="r", allow_pickle=False)
            if self.vectors.shape != (len(documents), DIMENSION):
                raise ValueError(f"Invalid shape of the vector index at {path}")
        # Short single queries are fast on the CPU even while a generator uses the GPU.
        self.model.to("cpu").float()
        self.query("поиск")

    @torch.inference_mode()
    def _encode(self, texts: list[str], device: str) -> np.ndarray:
        protect_tokenizer(self.tokenizer)
        batch = self.tokenizer(
            texts, max_length=TOKEN_LIMIT, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        hidden = self.model(**batch).last_hidden_state.float()
        mask = batch.attention_mask.unsqueeze(-1)
        means = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return torch.nn.functional.normalize(means, dim=1).cpu().numpy()

    def _build(self, documents: list[dict], path: Path) -> None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.model.to(device)
        if device != "cpu":
            self.model.half()
        messages = ["passage: " + semantic_text(doc["doc"][:2600]) for doc in documents]
        # Length bucketing reduces padding; output rows always retain corpus order.
        order = sorted(range(len(messages)), key=lambda i: len(messages[i]))
        fd, temporary = tempfile.mkstemp(prefix=path.stem, suffix=".tmp", dir=path.parent)
        os.close(fd)
        try:
            output = np.lib.format.open_memmap(
                temporary, mode="w+", dtype=np.float32, shape=(len(messages), DIMENSION)
            )
            batch_size = 48 if device != "cpu" else 12
            print(f"Building E5 index for {len(order)} messages on {device} …", flush=True)
            for offset in range(0, len(order), batch_size):
                rows = order[offset:offset + batch_size]
                output[rows] = self._encode([messages[i] for i in rows], device)
                if (offset // batch_size + 1) % 50 == 0:
                    print(f"E5: {offset + len(rows)}/{len(order)}", flush=True)
            output.flush()
            del output
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def memory_index(self, documents: list[dict]) -> DenseIndex:
        """Encode a visitor's inbox using the resident CPU model, without writing vectors to disk."""
        index = copy(self)
        index.vectors = np.empty((len(documents), DIMENSION), dtype=np.float32)
        for start in range(0, len(documents), 12):
            texts = ["passage: " + semantic_text(doc["doc"][:2600]) for doc in documents[start:start + 12]]
            with self._query_lock:
                index.vectors[start:start + len(texts)] = self._encode(texts, "cpu")
        return index

    def query(self, text: str) -> np.ndarray:
        with self._query_lock:
            return self._encode(["query: " + text], "cpu")[0]


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Prepare the multilingual E5 message index.")
    parser.add_argument("--documents", type=Path, default=Path("data/documents.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/mailsearch"))
    args = parser.parse_args()
    with args.documents.open(encoding="utf-8") as source:
        index = DenseIndex([json.loads(line) for line in source if line.strip()], args.cache_dir)
    print(f"E5 index ready: {index.vectors.shape[0]} messages", flush=True)
