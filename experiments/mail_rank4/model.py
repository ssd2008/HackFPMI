"""Trainable field fusion for natural-language known-item searches."""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from mailsearch.dense import DenseIndex
from mailsearch.dates import parse_date_scope
from mailsearch.store import CACHE, describe

COMMAND = re.compile(r"^\s*(?:покажи|найди|отыщи|найти)\s+", re.I)
PREFIX = re.compile(
    r"^(?:(?:мне|пожалуйста|то|там|где|про|о том как|письмо|сообщение|уведомление)\s+)*",
    re.I,
)
STRUCTURED = re.compile(
    r'[@«»"“”]|\b(?:артикул\w*|домен\w*|отправител\w*|с темой|со слов\w*|письма от|письмо от|за период|тема:|syn-|l-|\d{5,})',
    re.I,
)


def content(query):
    return PREFIX.sub("", COMMAND.sub("", query), count=1).strip()


def titles(docs):
    return [dict(doc_id=d["doc_id"], doc=describe(d)["subject"]) for d in docs]


class Signals:
    def __init__(self, docs, dense):
        self.dense = dense
        heading = titles(docs)
        self.headings = DenseIndex(heading, CACHE).vectors
        self.encoder = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=110000,
            dtype=np.float32,
        )
        texts = [
            h["doc"] + " " + h["doc"] + " " + d["doc"][:1500]
            for h, d in zip(heading, docs)
        ]
        self.chars = (
            self.encoder.fit_transform(texts) if any(t.strip() for t in texts) else None
        )

    def features(self, query, indices):
        vector = self.dense.query(query)
        char = (
            (self.chars[indices] @ self.encoder.transform([query]).T).toarray().ravel()
            if self.chars is not None
            else np.zeros(len(indices))
        )
        return np.column_stack(
            (
                self.dense.vectors[indices] @ vector,
                self.headings[indices] @ vector,
                char,
            )
        )


class ParaphraseSearch:
    def __init__(self, base, docs, dense, weights):
        self.base = base
        self.docs = docs
        self.boxes = np.array([d["synthetic_user"] for d in docs])
        self.dates = np.array([d["doc_dt"] or "" for d in docs])
        self.signals = Signals(docs, dense)
        self.weights = weights

    def search(self, mailbox, query, at, limit):
        if (
            not COMMAND.match(query)
            or STRUCTURED.search(query)
            or parse_date_scope(query, at).bounded
            or len(content(query).split()) < 3
        ):
            return self.base.search(mailbox, query, at, limit)
        old = self.base.search(mailbox, query, at, max(limit, 32))
        idx = np.flatnonzero((self.boxes == mailbox) & (self.dates <= at.isoformat()))
        if not len(idx):
            return []
        x = self.signals.features(content(query), idx)
        if not old and max(x[:, :2].max(axis=1)) < 0.82:
            return []
        scores = x @ self.weights
        order = np.argsort(-scores, kind="stable")[: max(64, limit)]
        # Reciprocal rank fusion retains the field-aware search as a secondary signal.
        blended = {
            self.docs[idx[i]]["doc_id"]: 1 / (8 + pos) for pos, i in enumerate(order)
        }
        for pos, (did, _) in enumerate(old):
            blended[did] = blended.get(did, 0) + 0.28 / (8 + pos)
        return sorted(blended.items(), key=lambda pair: -pair[1])[:limit]
