"""Average field and token rank votes while keeping the candidate pool and total vote fixed."""

import numpy as np

from experiments.mail_rank4.model import content
from .interaction import InteractionSearch
from .reranker import build_candidate, reciprocal


class JointSearch:
    def __init__(self, bilingual, docs):
        self.bilingual = bilingual
        self.field = build_candidate("trained", bilingual, docs)
        self.tokens = InteractionSearch(bilingual, docs)

    def search(self, mailbox, query, at, limit):
        candidates = self.field._pool(mailbox, query, at, limit)
        if candidates is None:
            return self.bilingual.search(mailbox, query, at, limit)
        rows, scores, semantic = candidates
        if not len(rows):
            return []
        text = content(query)
        features = self.field.encoder.features(text, rows, semantic)
        interactions = self.tokens.document_index.features(text, rows)
        votes = reciprocal(len(rows), 1.5, 8)
        for values in (features @ self.field.coef, interactions @ self.tokens.coef):
            ranking = np.argsort(-values, kind="stable")
            scores[ranking] += votes
        order = np.argsort(-scores, kind="stable")[:limit]
        return [(str(self.bilingual.ids[rows[i]]), float(scores[i])) for i in order]
