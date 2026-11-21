"""Compare two /api/search revisions on official dev and the author's fixed OOD panels."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
from time import perf_counter_ns
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import check_service_corpus, fetch, load_benchmark, metrics, percentiles
from mailsearch.dates import parse_date_scope
from tools.check_mail_ood import SCOPE, check_inputs, commit_id, digest, lines, measure
from tools.validate_rankings import keyed, load_labels, prepare_tasks

SIDES = ("before", "after")
K = 10


@dataclass
class Workload:
    name: str
    docs: dict
    queries: dict
    labels: dict
    metadata: dict
    tasks: dict
    hashes: dict

    def describe(self):
        counts = [sum(grade > 0 for grade in rel.values()) for rel in self.labels.values()]
        positive = [n for n in counts if n]
        return {"queries": len(counts), "positive": len(positive), "single_positive": counts.count(1),
                "verified_empty": counts.count(0), "known_item_targets": sum(t.kind == "known_item" for t in self.tasks.values()),
                "declared_windows": sum(bool(t.get("receipt_window")) for t in self.metadata.values()),
                "recall_ceiling_at_10": statistics.mean(min(K, n) / n for n in positive), "sha256": self.hashes}


def official(data):
    docs, rows, labels, dataset = load_benchmark(data)
    queries = keyed(rows, "official queries")
    meta = {}
    for qid, q in queries.items():
        positive = [d for d, g in labels[qid].items() if g > 0]
        days = {docs[d][1] for d in positive}
        span = parse_date_scope(q["query"], date.fromisoformat(q["query_date"]))
        # Confirm the explicitly requested day against the independent positive qrels.
        if not span.start or span.start != span.end or days != {span.start.isoformat()}:
            raise ValueError("Expected the official single-receipt-day benchmark.")
        day = span.start.isoformat()
        meta[qid] = {"query_id": qid, "family_id": qid, "source": "vk", "split": "dev",
                     "state": "positive", "task_type": "known_item" if len(positive) == 1 else "collection",
                     "target_ids": positive if len(positive) == 1 else [], "receipt_window": [day, day]}
    if len(queries) != 99:
        raise ValueError("Expected 99 official dev queries.")
    return Workload("official_vk", docs, queries, labels, meta, prepare_tasks(list(meta.values()), queries, labels),
                    dataset["sha256"])


def panel(pool, source, split="dev"):
    name = f"{source}_{split}"
    folder = pool / "suite" / name
    path = pool / "panels" / (name + ".jsonl")
    manifest = check_inputs(pool, name)
    ids = keyed(lines(path), "fixed panel")
    def selected(filename):
        return [r for r in lines(folder / filename) if r["query_id"] in ids]
    queries = keyed(selected("queries_dev.jsonl") + selected("queries_empty.jsonl"), "panel queries")
    labels = load_labels(selected("qrels_dev.jsonl") + selected("qrels_empty.jsonl"), queries)
    metadata = keyed(selected("cases.jsonl"), "panel tasks")
    if len(ids) != 250 or queries.keys() != ids.keys() or metadata.keys() != ids.keys():
        raise ValueError("The fixed panel and suite queries differ.")
    docrows = lines(folder / "documents.jsonl")
    docs = {r["doc_id"]: (r["synthetic_user"], r["doc_dt"]) for r in docrows}
    if len(docs) != len(docrows) or len(docs) != manifest["documents"]:
        raise ValueError("The panel corpus contains duplicate or missing documents.")
    for qid, rel in labels.items():
        q = queries[qid]
        span = metadata[qid].get("receipt_window")
        if span and (len(span) != 2 or not span[0] <= span[1] <= q["query_date"]):
            raise ValueError("Invalid panel receipt window.")
        for did, grade in rel.items():
            mailbox, received = docs[did]
            if mailbox != q["synthetic_user"] or (received and received > q["query_date"]):
                raise ValueError("A panel judgment is outside the request scope.")
            if span and grade > 0 and (not received or not span[0] <= received <= span[1]):
                raise ValueError("A positive panel judgment is outside the requested receipt window.")
    return Workload(name, docs, {qid: queries[qid] for qid in ids}, labels, metadata,
                    prepare_tasks(list(metadata.values()), queries, labels),
                    {"documents.jsonl": manifest["sha256"]["documents.jsonl"], "manifest": digest(folder / "manifest.json"),
                     "panel": digest(path)})


def ordered_passes(ids, passes, seed):
    rng = random.Random(seed)
    order = []
    for _ in range(passes):
        order.extend(rng.sample(ids, len(ids)))
    return order


def timed_series(url, work, ids, clients, reference, scope_by_ranking, first_measured=None):
    def request(qid):
        start = perf_counter_ns()
        ranked, server_ms, _ = fetch(url, work.queries[qid], K)
        duration = (perf_counter_ns() - start) / 1_000_000
        return qid, tuple(ranked), duration, server_ms

    wall, server = [], []
    changes = 0
    with ThreadPoolExecutor(max_workers=clients) as pool:
        for qid, ranked, elapsed, backend in pool.map(request, ids):
            if len(ranked) > K or len(set(ranked)) != len(ranked):
                raise ValueError("An endpoint returned duplicate results or exceeded K.")
            if reference.setdefault(qid, ranked) != ranked:
                changes += 1
            key = (qid, ranked)
            if key not in scope_by_ranking:
                _, scope_by_ranking[key] = measure(ranked, work.tasks[qid], work.labels[qid],
                                                   work.metadata[qid], work.queries[qid], work.docs)
            if first_measured is not None:
                first_measured.setdefault(qid, ranked)
            wall.append(elapsed)
            server.append(backend)
    return wall, server, changes


def quality(work, ranks):
    values = []
    scope = Counter()
    for qid, q in work.queries.items():
        row, checked = measure(ranks[qid], work.tasks[qid], work.labels[qid], work.metadata[qid], q, work.docs)
        if sum(g > 0 for g in work.labels[qid].values()) == 1:
            row["singleton_rr"] = metrics(ranks[qid], work.labels[qid], K)["mrr_single"]
        values.append(row)
        scope.update(checked)
    def mean(key):
        subset = [r[key] for r in values if key in r]
        return round(statistics.mean(subset), 8) if subset else None
    return {"recall_at_10_positive": mean("known_recall_at_10"), "ndcg_at_10_positive": mean("ndcg_at_10"),
            "mrr_at_10_single": mean("singleton_rr"), "mrr_queries": sum("singleton_rr" in r for r in values),
            "author_target_mrr_at_10": mean("mrr_at_10"), "author_target_queries": sum("mrr_at_10" in r for r in values),
            "false_empty": sum(r.get("false_empty_rate", 0) for r in values),
            "false_nonempty": sum(r.get("false_nonempty_rate", 0) for r in values),
            "scope_on_unique_queries": {k: scope[k] for k in SCOPE}}


def evaluate(work, urls, args, dataset_index):
    result = {"dataset": work.describe(), "server_corpus_check": {}, "by_clients": {}}
    reference = {s: {} for s in SIDES}
    first = {s: {} for s in SIDES}
    scoped = {s: {} for s in SIDES}
    for side in SIDES:
        result["server_corpus_check"][side] = check_service_corpus(urls[side], work.docs, work.hashes["documents.jsonl"])
    ids = list(work.queries)
    passes = math.ceil(args.min_samples / len(ids))
    for mode_index, clients in enumerate(args.clients):
        seed = args.seed + dataset_index * 10000 + clients * 100
        warm = ordered_passes(ids, 1, seed)
        combined = {s: {"wall": [], "server": []} for s in SIDES}
        starter = (dataset_index + mode_index) % 2
        warm_order = SIDES if starter == 0 else SIDES[::-1]
        warm_changes = {s: timed_series(urls[s], work, warm, clients, reference[s], scoped[s])[2] for s in warm_order}
        pairs = []
        for i in range(args.pairs):
            order = SIDES if (starter + i) % 2 == 0 else SIDES[::-1]
            queries = ordered_passes(ids, passes, seed + i + 1)
            pair = {"pair": i + 1, "order": list(order), "order_seed": seed + i + 1}
            for side in order:
                wall, server, changes = timed_series(urls[side], work, queries, clients, reference[side],
                                                     scoped[side], first[side])
                combined[side]["wall"].extend(wall)
                combined[side]["server"].extend(server)
                pair[side] = {"requests": len(wall), "client_http_ms": percentiles(wall),
                              "server_ms": percentiles(server), "changed_rankings": changes}
            pair["delta_client_p99_ms"] = round(pair["after"]["client_http_ms"]["p99"] - pair["before"]["client_http_ms"]["p99"], 3)
            pair["delta_server_p99_ms"] = round(pair["after"]["server_ms"]["p99"] - pair["before"]["server_ms"]["p99"], 3)
            pairs.append(pair)
            print(f"{work.name}, {clients} clients, pair {i + 1}/{args.pairs} complete", flush=True)
        result["by_clients"][str(clients)] = {
            "warm_order": list(warm_order), "warm_queries_per_side": len(warm), "warm_changed_rankings": warm_changes,
            "requests_per_side": passes * len(ids) * args.pairs, "pairs": pairs,
            "pooled": {s: {"client_http_ms": percentiles(b["wall"]), "server_ms": percentiles(b["server"]),
                           "median_of_pair_client_p99_ms": round(statistics.median(p[s]["client_http_ms"]["p99"] for p in pairs), 3)}
                       for s, b in combined.items()}}
    result["quality"] = {s: quality(work, first[s]) for s in SIDES}
    def paired(a, b):
        return {"before": a, "after": b, "delta_after_minus_before": round(b - a, 8) if None not in (a, b) else None}
    qa, qb = (result["quality"][s] for s in SIDES)
    result["four_metrics_by_clients"] = {
        setting: {**{k: paired(qa[k], qb[k]) for k in ("recall_at_10_positive", "ndcg_at_10_positive", "mrr_at_10_single")},
                  "server_p99_ms": paired(block["pooled"]["before"]["server_ms"]["p99"],
                                          block["pooled"]["after"]["server_ms"]["p99"]),
                  "client_p99_ms": paired(block["pooled"]["before"]["client_http_ms"]["p99"],
                                         block["pooled"]["after"]["client_http_ms"]["p99"])}
        for setting, block in result["by_clients"].items()}
    result["within_version_changed_rankings"] = {
        s: sum(block["warm_changed_rankings"][s] + sum(pair[s]["changed_rankings"] for pair in block["pairs"])
               for block in result["by_clients"].values()) for s in SIDES}
    result["scope_across_distinct_query_rankings"] = {
        s: {"distinct_query_rankings": len(rows), **{k: sum(row[k] for row in rows.values()) for k in SCOPE}}
        for s, rows in scoped.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--public-data", type=Path, default=ROOT / "data")
    parser.add_argument("--include-holdout", action="store_true", help="Include both supplied holdout panels.")
    parser.add_argument("--pairs", type=int, choices=(2, 3), default=2)
    parser.add_argument("--clients", type=int, nargs="+", choices=(1, 8), default=[1, 8])
    parser.add_argument("--min-samples", type=int, default=1000, help="Min requests per side, pair, suite and client count.")
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and print denominators without opening a connection.")
    parser.add_argument("--output", type=Path)
    for side in SIDES:
        parser.add_argument(f"--{side}-revision")
        for src in ("vk", "personal"):
            parser.add_argument(f"--{src}-{side}")
    args = parser.parse_args()
    if args.min_samples < 1 or len(set(args.clients)) != len(args.clients):
        parser.error("Choose a positive sample count and distinct client settings.")
    splits = ("dev", "holdout") if args.include_holdout else ("dev",)
    workloads = [official(args.public_data), *(panel(args.pool, src, split) for split in splits for src in ("vk","personal"))]
    if args.dry_run:
        print(json.dumps({w.name: w.describe() for w in workloads}, indent=2))
        return
    versions = {s: getattr(args, s + "_revision") for s in SIDES}
    urls = {src: {s: getattr(args, src + "_" + s) for s in SIDES} for src in ("vk", "personal")}
    if not args.output or not all(versions.values()) or not all(
        u and urlsplit(u).scheme == "http" and urlsplit(u).hostname in {"localhost", "127.0.0.1", "::1"}
        for pair in urls.values() for u in pair.values()
    ):
        parser.error("Supply an output, both pinned revision labels and four loopback HTTP URLs.")
    report = {"revisions": {s: commit_id(v) for s, v in versions.items()},
              "server_ports": {src: {s: urlsplit(u).port or 80 for s, u in sides.items()} for src, sides in urls.items()},
              "k": K, "pairs": args.pairs, "clients": args.clients, "seed": args.seed,
              "min_samples_per_side_per_pair": args.min_samples, "platform": platform.platform(),
              "python": platform.python_version(), "protocol": {
                  "latency": "Closed-loop workers. The primary client clock is monotonic perf_counter_ns around payload construction, HTTP, full body read and JSON decode. Server-Timing is reported separately. The four-metric comparison uses pooled request p99, with per-pair p99 and its median alongside. Any HTTP or malformed-response error aborts the measurement; missing samples are never excluded silently. All percentiles use nearest rank. Each series includes whole shuffled passes. Identical dispatch order for both sides in a pair; the second side starts only after the first completes.",
                  "quality": "Query-macro averages on the first measured response, counted once per query. Recall/nDCG use graded positives and primary MRR uses queries with exactly one positively graded document; the separate author-target MRR uses designated known-item targets. Verified empty queries are counted separately. Recall ceiling is mean(min(10,n)/n) on positives. The official receipt day is confirmed by the positive qrels; OOD windows are supplied by the author.",
                  "stability": "Every ranked list including subsequent warmups is compared with that version's first warm response across all client counts. Each distinct ordered query result is checked for corpus membership, mailbox, as-of date and declared receipt window. Different revisions may return different rankings.",
                  "provenance": "Revision labels resolve to local Git commits. The service corpus is checked over HTTP; associating a running service with the intended commit relies on launching that checkout on its declared port.",
                  "privacy": "Only aggregates and input file digests are published; no query text, returned IDs or mailbox information."},
              "workloads": {}}
    for i, w in enumerate(workloads):
        report["workloads"][w.name] = evaluate(w, urls["personal" if w.name.startswith("personal_") else "vk"], args, i)
    stable = all(not any(r["within_version_changed_rankings"].values()) for r in report["workloads"].values())
    report["status"] = "complete" if stable else "unstable_rankings"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    if not stable:
        raise SystemExit("The report was saved. At least one revision changed its ranking between repeated requests.")


if __name__ == "__main__":
    main()
