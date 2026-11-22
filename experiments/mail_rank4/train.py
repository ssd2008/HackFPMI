"""Fit fusion weights on positive vs negative passages from external train only."""

from collections import defaultdict
import json
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from mailsearch.dense import DenseIndex
from mailsearch.store import CACHE, ROOT, load_jsonl
from .model import Signals


def main():
    groups = defaultdict(list)
    for r in load_jsonl(ROOT / "datasets/local/mail-rank-train.jsonl"):
        groups[r["query"]].append(r)
    queries = random.Random(45201).sample(sorted(groups), 640)
    docs = []
    indices = {}
    for q in queries:
        indices[q] = []
        for p in groups[q]:
            indices[q].append((len(docs), p["label"]))
            docs.append(
                dict(
                    doc_id=str(len(docs)),
                    doc=f"Тема: {p['title']}\nТекст письма: {p['text']}",
                    doc_dt=None,
                )
            )
    signals = Signals(docs, DenseIndex(docs, CACHE))
    differences = []
    for q in queries:
        idx, labels = zip(*indices[q])
        x = signals.features(q, list(idx))
        positive = x[np.array(labels) == 1][0]
        differences.extend(positive - x[np.array(labels) == 0])
    diffs = np.asarray(differences)
    x = np.concatenate([diffs, -diffs])
    y = np.repeat([1, 0], len(diffs))
    scale = StandardScaler(with_mean=False).fit(x)
    fit = LogisticRegression(C=1, fit_intercept=False).fit(scale.transform(x), y)
    weights = (fit.coef_[0] / scale.scale_).tolist()
    (ROOT / "results/mail_rank4/fusion.json").write_text(
        json.dumps(
            dict(weights=weights, pairs=len(diffs), queries=len(queries), seed=45201),
            indent=2,
        )
        + "\n"
    )
    print("weights", weights, flush=True)


if __name__ == "__main__":
    main()
