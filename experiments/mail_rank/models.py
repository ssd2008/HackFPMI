"""Second-stage lexical interaction models over the project's hybrid shortlist."""
import re
import threading
from functools import lru_cache

import joblib
import numpy as np
import snowballstemmer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TOKEN = re.compile(r"[\w]+(?:[-@.][\w]+)*", re.UNICODE)
_thread = threading.local()


@lru_cache(maxsize=64000)
def stem(token):
    if not token.isalpha() or token.isascii():
        return token
    if not hasattr(_thread, "stemmer"):
        _thread.stemmer = snowballstemmer.stemmer("russian")
    return _thread.stemmer.stemWord(token)


def representation(text):
    words = tuple(TOKEN.findall(text.lower().replace("ё", "е")))
    stems = tuple(stem(t) for t in words)
    grams = {w[i:i+3] for w in words for i in range(len(w)-2)}
    return (set(words), set(stems), grams, set(zip(stems, stems[1:])),
            {w for w in words if any(c.isdigit() for c in w)})


def overlap(query, passage):
    values = []
    for q, d in zip(query, passage):
        n = len(q & d)
        values.extend((n / max(1, len(q)), n / max(1, len(q) * len(d)) ** .5))
    return values


def features(query, title, body):
    return overlap(query, title) + overlap(query, body)


def fit_models(rows, folder):
    x, y = [], []
    for row in rows:
        x.append(features(representation(row["query"]), representation(row["title"]),
                          representation(row["text"][:2200])))
        y.append(row["label"])
    candidates = {
        "linear": LogisticRegression(C=1, max_iter=300),
        "mlp": MLPClassifier(hidden_layer_sizes=(32, 12), max_iter=60,
                             random_state=2026, batch_size=256),
    }
    for name, classifier in candidates.items():
        model = make_pipeline(StandardScaler(), classifier).fit(x, y)
        joblib.dump(model, folder / (name + ".joblib"))


class Ranked:
    def __init__(self, engine, store, model):
        self.engine = engine
        self.store = store
        self.model = model
        self.document = lru_cache(maxsize=8192)(self._document)

    def _document(self, did):
        doc = self.store.by_id[did]
        return (representation(self.store.summaries[did]["subject"]),
                representation(doc["doc"][:2200]))

    def search(self, mailbox, query, at, limit):
        hits = self.engine.search(mailbox, query, at, max(32, limit))
        if len(hits) < 2:
            return hits[:limit]
        q = representation(query)
        matrix = [features(q, *self.document(did)) for did, _ in hits]
        probability = self.model.predict_proba(matrix)[:, 1]
        # Preserve the baseline's absolute score differences and literal constraints.
        ranked = [(did, score + .25 * float(p)) for (did, score), p in zip(hits, probability)]
        return sorted(ranked, key=lambda item: -item[1])[:limit]


if __name__ == "__main__":
    import json
    from pathlib import Path

    out = Path("results/mail_rank")
    out.mkdir(parents=True, exist_ok=True)
    with Path("datasets/local/mail-rank-train.jsonl").open() as stream:
        fit_models(map(json.loads, stream), out)
