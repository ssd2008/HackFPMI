"""Rerank the union of incumbent, E5 and translated lexical candidates."""
from collections import Counter
from functools import lru_cache
import json

import numpy as np

from experiments.mail_rank4.bilingual import words
from experiments.mail_rank4.model import COMMAND, STRUCTURED, content
from mailsearch.dates import parse_date_scope
from mailsearch.retrieval import _CODE_CANDIDATE, _article
from mailsearch.store import ROOT, describe

WEIGHTS = ROOT / "results/recall_cycle/learned.json"
FEATURES = ("e5", "exact_body", "translated_body", "exact_title", "translated_title",
            "second_match", "adjacent_pair")


def natural(query, at):
    return (COMMAND.match(query) and not STRUCTURED.search(query)
            and len(content(query).split()) >= 3 and not parse_date_scope(query, at).bounded
            and not any(not m.lastgroup and _article(m[0]) for m in _CODE_CANDIDATE.finditer(query)))


class FieldFeatures:
    def __init__(self, docs, terms):
        self.docs, self.terms = docs, terms
        self.fields = lru_cache(maxsize=10000)(self._fields)
        freq = Counter()
        for doc in docs:
            fields = describe(doc)
            freq.update(set(words(fields["subject"] + " " + doc["doc"][:2100])))
        self.idf = {t: np.log1p(len(docs) / (n + 1)) for t, n in freq.items()}
        self.max_idf = np.log1p(len(docs) / 2)

    def _fields(self, row):
        doc = self.docs[row]
        summary = describe(doc)
        title = set(words(summary["subject"] + " " + (summary["sender"] or "")))
        body_tokens = words(doc["doc"][:2100])
        pairs = set(zip(body_tokens, body_tokens[1:]))
        return title, set(body_tokens), pairs

    def features(self, query, rows, semantic):
        tokens = list(dict.fromkeys(words(query)))
        candidates = []
        weights = []
        for t in tokens:
            choices = [(t, 1.)] + [(en, val) for en, val in self.terms.get(t, []) if en != t]
            candidates.append(choices)
            # A query concept missing in the corpus can still have common translations.
            wt = min((self.idf.get(w, self.max_idf) for w, v in choices if v >= .2), default=self.max_idf)
            weights.append(max(.2, wt))
        norm = sum(weights) or 1
        seq = list(zip(tokens, tokens[1:]))
        X = np.zeros((len(rows), len(FEATURES)), dtype=np.float32)
        X[:, 0] = semantic
        for n, row in enumerate(rows):
            title, body, pairs = self.fields(int(row))
            maxima = []
            for token, choices, weight in zip(tokens, candidates, weights):
                body_strength = max((v for w, v in choices if w in body), default=0.)
                title_strength = max((v for w, v in choices if w in title), default=0.)
                if token in body:
                    X[n, 1] += weight
                if token in title:
                    X[n, 3] += weight
                X[n, 2] += weight * body_strength
                X[n, 4] += weight * title_strength
                maxima.append(max(body_strength, title_strength))
            X[n, 1:5] /= norm
            X[n, 5] = sorted(maxima, reverse=True)[1] if len(maxima) > 1 else 0.
            X[n, 6] = sum(p in pairs for p in seq) / max(1, len(seq))
        return X


def reciprocal(ranks, weight=1., pivot=10):
    return weight / (pivot + np.arange(ranks, dtype=np.float32))


class Reranker:
    def __init__(self, variant, baseline, docs):
        self.variant, self.baseline = variant, baseline
        self.indices = {d["doc_id"]: i for i, d in enumerate(docs)}
        if variant == "trained":
            weights = json.loads(WEIGHTS.read_text())
            if tuple(weights["features"]) != FEATURES:
                raise ValueError("Wrong feature contract")
            self.coef = np.array(weights["coef"])
            self.encoder = FieldFeatures(docs, baseline.terms)

    def _pool(self, mailbox, query, at, limit):
        base = self.baseline
        if base.matrix is None or not natural(query, at):
            return None
        ix = np.flatnonzero((base.boxes == mailbox) & (base.dates <= at.isoformat()))
        if not len(ix):
            return ix, np.empty(0), np.empty(0)
        text = content(query)
        # Bilingual internally fuses at least 40 candidates for every small limit.
        # Preserve that shortlist, including its tail, before adding new candidates.
        old = base.search(mailbox, query, at, max(limit, 40))
        if not old:
            return ix[:0], np.empty(0), np.empty(0)
        vector = base.dense.query(text)
        semantic = base.dense.vectors[ix] @ vector
        tokens = words(text)
        translated = {}
        for t in tokens:
            for en, v in base.terms.get(t, []):
                translated[en] = max(translated.get(en, 0.), v)
        translated.update(dict.fromkeys(tokens, 1.))
        vocab = base.encoder.vocabulary_
        q = np.zeros(len(vocab), dtype=np.float32)
        for t, v in translated.items():
            if t in vocab:
                j = vocab[t]
                q[j] = v * base.encoder.idf_[j]
        lex = base.matrix[ix] @ q
        mixed = lex / max(1.e-6, lex.max()) + np.clip((semantic - .73) / .16, 0, 1.5)
        score = Counter()
        source_depth = max(100, limit)
        for signal, positive, strength in ((lex, lex > 0, .5), (semantic, None, .8),
                                            (mixed, lex > 0, .9)):
            order = np.argsort(-signal, kind="stable")
            if positive is not None:
                order = order[positive[order]]
            ids = ix[order[:source_depth]]
            score.update(dict(zip(map(int, ids), reciprocal(len(ids), strength))))
        old_rows = [self.indices[d] for d, _ in old]
        score.update(dict(zip(old_rows, reciprocal(len(old_rows), 2.))))
        rows = np.array(sorted(score, key=lambda i: -score[i]), dtype=np.intp)
        return rows, np.array([score[int(i)] for i in rows]), semantic[np.searchsorted(ix, rows)]

    def pool(self, mailbox, query, at, limit):
        result = self._pool(mailbox, query, at, limit)
        return None if result is None else result[:2]

    def search(self, mailbox, query, at, limit):
        result = self._pool(mailbox, query, at, limit)
        if result is None:
            return self.baseline.search(mailbox, query, at, limit)
        rows, scores, semantic = result
        if self.variant == "trained" and len(rows):
            X = self.encoder.features(content(query), rows, semantic)
            ranking = np.argsort(-(X @ self.coef), kind="stable")
            # Blend independent relevance evidence with the complete pool prior.
            scores[ranking] += reciprocal(len(ranking), 3., 8)
        order = np.argsort(-scores, kind="stable")[:limit]
        return [(str(self.baseline.ids[rows[i]]), float(scores[i])) for i in order]


def build_candidate(variant, baseline, docs):
    if variant not in ("trained", "pool"):
        raise ValueError(f"Unrecognised variant {variant}")
    return Reranker(variant, baseline, docs)
