"""Measure candidate recall before the hybrid ranker's proximity and decision stages.

The evaluation runs an isolated engine without an HTTP endpoint. It observes the
actual ordered rows fed to LexicalIndex.proximity. Direct-ID, date-only and other
early returns have no reranking stage and expose their already ordered matches.
"""
import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
from statistics import mean
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import load_benchmark
from mailsearch.retrieval import SearchEngine
from mailsearch.store import CACHE, DATA, load_corpus


def measure(args):
    _, queries, labels, dataset = load_benchmark(args.data, args.queries, args.qrels)
    docs, _ = load_corpus(args.data / "documents.jsonl")
    engine = SearchEngine(docs, CACHE)
    original = engine.lexical.proximity
    observed = []

    def probe(features, candidates):
        observed.append([engine.ids[i] for i in candidates])
        return original(features, candidates)

    modes = Counter()
    reference = None
    if args.final_report:
        report = json.loads(args.final_report.read_bytes())
        if dataset["sha256"] != report["dataset"]["sha256"]:
            raise ValueError("Recorded rankings and diagnostic inputs differ")
        reference = report["queries"]
    details = {}
    with patch.object(engine.lexical, "proximity", side_effect=probe):
        for q in queries:
            query_id = q["query_id"]
            positive = {doc_id for doc_id, grade in labels[query_id].items() if grade > 0}
            if not positive:
                continue
            observed.clear()
            hits = engine.search(q["synthetic_user"], q["query"], date.fromisoformat(q["query_date"]), limit=128)
            final = [doc_id for doc_id, _ in hits]
            if reference is not None and final[:10] != reference[query_id]["ranked_ids"][:10]:
                raise ValueError("Top 10 does not match the real search endpoint for " + query_id)
            if len(observed) > 1:
                raise ValueError("The search implementation has more than one proximity stage")
            raw = observed[0] if observed else final
            stage = "before_proximity" if observed else "direct_or_no_candidates"
            modes[stage] += 1
            details[query_id] = {
                "path": stage,
                "candidate_count": len(raw),
                "known_positives": len(positive),
                "candidate_recall_50": len(positive.intersection(raw[:50])) / len(positive),
                "candidate_recall_128": len(positive.intersection(raw[:128])) / len(positive),
                "final_recall_10": len(positive.intersection(final[:10])) / len(positive),
            }
    if not details:
        raise ValueError("No queries with positive judgments")
    metrics = ("candidate_recall_50", "candidate_recall_128", "final_recall_10")
    result = {
        "dataset": dataset,
        "known_positive_queries": len(details),
        "paths": dict(modes),
        "compared_to_http_top_10": reference is not None,
        "method": ("Actual first-stage rows in initial score order before passage proximity, final evidence "
                   "threshold and status filters. Only the top 128 receive the proximity adjustment. Other "
                   "rows can still qualify on the base score. Direct lookups have no reranker."),
        "mean": {metric: round(mean(item[metric] for item in details.values()), 6) for metric in metrics},
        "queries": details,
    }
    print(json.dumps({key: result[key] for key in ("known_positive_queries", "paths", "mean")}, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--qrels", type=Path)
    parser.add_argument("--final-report", type=Path, help="Check top-10 parity against benchmark.py output.")
    parser.add_argument("--output", type=Path)
    measure(parser.parse_args())
