"""Reproduce the team's public character baseline and separate easy and hard dates."""

import argparse
from collections import defaultdict
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
import sys

import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import load_benchmark, metrics
from mailsearch.store import DATA, load_jsonl
from tools.validate_rankings import read_report


def evaluate(before_file, after_file):
    _, queries, labels, dataset = load_benchmark(DATA)
    documents = load_jsonl(DATA / "documents.jsonl")
    hashes = dataset["sha256"]
    by_query = {q["query_id"]: q for q in queries}
    ranks = {}
    for key, path in (("original_mvp", before_file), ("candidate", after_file)):
        ranking, metadata = read_report(path, by_query, labels, hashes["queries"], hashes["qrels"], dataset["qrels"])
        if metadata["documents_sha256"] != hashes["documents.jsonl"]:
            raise ValueError(f"Different corpus in {path.name}")
        ranks[key] = ranking

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 max_features=160000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(d["doc"] for d in documents)
    by_day = defaultdict(list)
    for i, doc in enumerate(documents):
        by_day[doc["synthetic_user"], doc["doc_dt"]].append(i)
    month_names = "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split()
    groups = defaultdict(list)
    for query in queries:
        qid = query["query_id"]
        intent, separator, day_expression = query["query"].partition(", полученные ")
        if not separator:
            raise ValueError("Unexpected official query format: " + qid)
        if day_expression.rstrip(".") == "вчера":
            day = date.fromisoformat(query["query_date"]) - timedelta(days=1)
        else:
            match = re.fullmatch(r"(\d{1,2}) (\w+) (\d{4}) года\.?", day_expression)
            if match is None:
                raise ValueError(f"Unexpected date in {qid}")
            day = date(int(match[3]), month_names.index(match[2]) + 1, int(match[1]))
        candidates = by_day[query["synthetic_user"], day.isoformat()]
        doc_ids = [documents[i]["doc_id"] for i in candidates]
        positives = {did for did, grade in labels[qid].items() if grade > 0}
        if not positives.issubset(doc_ids):
            raise ValueError(f"A public positive falls outside the day window in {qid}")
        subset = "day_equals_positives" if positives == set(doc_ids) else "content_required"
        rankers = {method: ranking[qid] for method, ranking in ranks.items()}
        rankers["date_only"] = sorted(doc_ids)
        for name, text in (("char_full_query", query["query"]), ("char_without_received_clause", intent)):
            scores = (vectorizer.transform([text]) @ matrix[candidates].T).toarray().ravel()
            rankers[name] = [doc_ids[i] for i in np.argsort(-scores, kind="stable")[:10]]
        row = {method: metrics(order, labels[qid], 10) for method, order in rankers.items()}
        groups["all"].append(row)
        groups[subset].append(row)

    slices = {}
    for name, rows in groups.items():
        summary = {}
        for method in rows[0]:
            summary[method] = {}
            for metric in ("ndcg", "recall", "mrr_single"):
                values = [row[method][metric] for row in rows if metric in row[method]]
                summary[method][metric] = round(mean(values), 6) if values else None
        slices[name] = {"queries": len(rows), "one_positive": sum("mrr_single" in r["date_only"] for r in rows),
                        "at_10": summary}
    # The team's report identifies this individual, potentially relevant unjudged letter.
    # Vary just this label in a sensitivity check and keep every official score intact.
    review_query = "H-cf13f62178c5b81140c8"
    unjudged = "L-9df7d161164c1b561c24"
    official = labels[review_query]
    sensitivity = {}
    for name, ranking in ranks.items():
        order = ranking[review_query]
        sensitivity[name] = {
            "rank": order.index(unjudged) + 1 if unjudged in order else None,
            "ndcg10_official": round(metrics(order, official, 10)["ndcg"], 6),
            "ndcg10_if_additional_positive": round(metrics(order, {**official, unjudged: 1}, 10)["ndcg"], 6),
        }
    return {
        "study": "docs/vk_mail_search_analysis.md, sections 5.3, 8.3 and 9.6",
        "corpus_sha256": hashes["documents.jsonl"],
        "libraries": {"numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "reports_sha256": {"before": hashlib.sha256(before_file.read_bytes()).hexdigest(),
                           "after": hashlib.sha256(after_file.read_bytes()).hexdigest()},
        "method": ("Every ranker is evaluated on the same official binary qrels. Both character runs fit the "
                   "same 3–5 character TF-IDF index to the full public corpus and retrieve inside the independently "
                   "parsed mailbox and receipt day. Only the trailing explicit receipt clause is removed in the "
                   "second run. The two application runs use their saved HTTP rankings."),
        "slices": slices,
        "possible_missing_label": {"query_id": review_query, "doc_id": unjudged,
                                   "judged_by_organizers": unjudged in official, "sensitivity": sensitivity},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.before, args.after)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, group in result["slices"].items():
        print(name, group["queries"], {k: v["ndcg"] for k, v in group["at_10"].items()})
