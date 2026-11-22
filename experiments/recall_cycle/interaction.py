"""Rerank the comparison pool using supervised E5 token-level matches."""
import json

import numpy as np

from mailsearch.store import ROOT
from experiments.mail_rank4.model import content
from .interaction_index import FEATURE_NAMES, TokenIndex
from .reranker import build_candidate, reciprocal

TOKEN_CACHE = ROOT / ".cache/recall_cycle/tokens"
WEIGHTS = ROOT / "results/recall_cycle/interaction-weights.json"


class InteractionSearch:
    def __init__(self, baseline, docs):
        self.baseline = baseline
        self.pool = build_candidate("pool", baseline, docs)
        weights = json.loads(WEIGHTS.read_text())
        if weights["features"] != FEATURE_NAMES:
            raise ValueError("Interaction feature schema changed")
        self.coef = np.array(weights["coef"], dtype=np.float32)
        self.document_index = TokenIndex(docs, baseline.dense, TOKEN_CACHE)

    def search(self, mailbox, query, at, limit):
        candidates = self.pool.pool(mailbox, query, at, limit)
        if candidates is None:
            return self.baseline.search(mailbox, query, at, limit)
        rows, scores = candidates
        if not len(rows):
            return []
        features = self.document_index.features(content(query), rows)
        order = np.argsort(-(features @ self.coef), kind="stable")
        # Match the field reranker's fusion exactly so the feature block is comparable.
        scores[order] += reciprocal(len(order), 3., 8)
        chosen = np.argsort(-scores, kind="stable")[:limit]
        return [(str(self.baseline.ids[rows[i]]), float(scores[i])) for i in chosen]
