"""Run the public mail retrieval benchmark entirely on the local machine."""

import argparse
from collections import defaultdict
from datetime import date
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import statistics
from time import perf_counter
from urllib.request import Request, urlopen

from mailsearch.dates import parse_date_scope

from .ranking import METHODS, Search


ROOT = Path(__file__).resolve().parents[1]


def load_data(folder):
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    records = {}
    files = {"documents": "documents.jsonl", "queries": "queries_dev.jsonl", "qrels": "qrels_dev.jsonl"}
    for key, filename in files.items():
        raw = (folder / filename).read_bytes()
        expected_hash = manifest.get("sha256", {}).get(filename)
        if expected_hash and hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError(f"Контрольная сумма не совпала: {filename}")
        rows = [json.loads(line) for line in raw.decode("utf-8").split("\n") if line.strip()]
        if key in manifest and len(rows) != manifest[key]:
            raise ValueError(f"Количество строк не совпало: {filename}")
        records[key] = rows

    documents_by_id = {doc["doc_id"]: doc for doc in records["documents"]}
    queries_by_id = {query["query_id"]: query for query in records["queries"]}
    if len(documents_by_id) != len(records["documents"]) or len(queries_by_id) != len(records["queries"]):
        raise ValueError("Повторяющийся ID письма или запроса")
    gold = defaultdict(dict)
    for row in records["qrels"]:
        qid, did = row["query_id"], row["doc_id"]
        query, doc = queries_by_id[qid], documents_by_id[did]
        if doc["synthetic_user"] != query["synthetic_user"] or (doc["doc_dt"] or "") > query["query_date"]:
            raise ValueError(f"Разметка ссылается на недоступное письмо: {qid} / {did}")
        if row["relevance"] > 0:
            gold[qid][did] = row["relevance"]
    if any(not gold[qid] for qid in queries_by_id):
        raise ValueError("У каждого запроса должно быть хотя бы одно размеченное релевантное письмо")

    queries = [(q, date.fromisoformat(q["query_date"])) for q in records["queries"]]
    return records["documents"], queries, gold


def rewrite_queries(queries, path):
    replacements = json.loads(path.read_text(encoding="utf-8"))
    rewritten = []
    for query, as_of in queries:
        text = query["query"]
        matches = [(old, new) for old, new in replacements.items() if old in text]
        if len(matches) != 1:
            raise ValueError(f"Для запроса {query['query_id']} нужна ровно одна переформулировка")
        old, new = matches[0]
        rewritten.append(({**query, "query": text.replace(old, new, 1)}, as_of))
    return rewritten


def api_rank(endpoint, query, mailbox, as_of, k):
    body = json.dumps({"query": query, "mailbox": mailbox, "date": as_of.isoformat(), "limit": k})
    request = Request(endpoint, body.encode("utf-8"), {"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=20) as response:
        timing = re.search(r"\bsearch;dur=([\d.]+)", response.headers.get("Server-Timing", ""))
        payload = json.load(response)
    return [item["id"] for item in payload["items"]], float(timing[1]) if timing else None


def local_rank(search, method, query, mailbox, as_of, k):
    return search.rank(method, query, mailbox, as_of, k), None


def dcg(relevances):
    return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances))


def percentiles(samples):
    ordered = sorted(samples)
    return {f"p{p}": ordered[math.ceil(len(ordered) * p / 100) - 1] for p in (50, 95, 99)}


def evaluate(rank, queries, documents_by_id, gold, cutoffs, repeats):
    kmax = max(cutoffs)
    for query, as_of in queries[:5]:
        rank(query["query"], query["synthetic_user"], as_of, kmax)

    durations = []
    server_durations = []
    rankings = []
    for repeat in range(repeats):
        for query, as_of in queries:
            start = perf_counter()
            docs, server_elapsed = rank(query["query"], query["synthetic_user"], as_of, kmax)
            elapsed = (perf_counter() - start) * 1000
            durations.append(elapsed)
            if server_elapsed is not None:
                server_durations.append(server_elapsed)
            if repeat == 0:
                rankings.append((docs, elapsed, server_elapsed))

    details = []
    values = {k: defaultdict(list) for k in cutoffs}
    for (query, _), (docs, elapsed, server_elapsed) in zip(queries, rankings, strict=True):
        relevant = gold[query["query_id"]]
        if len(docs) != len(set(docs)) or len(docs) > kmax:
            raise ValueError(f"Некорректная выдача для {query['query_id']}")
        for doc_id in docs:
            doc = documents_by_id[doc_id]
            if doc["synthetic_user"] != query["synthetic_user"] or (doc["doc_dt"] or "") > query["query_date"]:
                raise ValueError(f"Недоступное письмо в выдаче: {query['query_id']} / {doc_id}")
        grades = [relevant.get(doc, 0) for doc in docs]
        ideal = sorted(relevant.values(), reverse=True)
        first = next((n for n, rel in enumerate(grades, 1) if rel), None)
        per_k = {}
        for k in cutoffs:
            metrics = {
                "Recall": sum(rel > 0 for rel in grades[:k]) / len(relevant),
                "nDCG": dcg(grades[:k]) / dcg(ideal[:k]),
            }
            if len(relevant) == 1:
                metrics["MRR-single"] = 1 / first if first and first <= k else 0.0
            per_k[k] = metrics
            for key, score in metrics.items():
                values[k][key].append(score)
        details.append({
            **query, "first_rel_rank": first, "ms": elapsed, "metrics": per_k, "doc_ids": docs,
            "missed_doc_ids": sorted(relevant.keys() - set(docs)),
        })
        if server_elapsed is not None:
            details[-1]["server_ms"] = server_elapsed
    summary = {k: {key: statistics.mean(scores) for key, scores in block.items()} for k, block in values.items()}
    result = {"metrics": summary, "latency_ms": percentiles(durations), "requests": len(durations), "per_query": details}
    if server_durations and len(server_durations) == len(durations):
        result["server_latency_ms"] = percentiles(server_durations)
    return result


def scope_stats(search, documents, queries, gold, cutoffs):
    all_sizes, scoped_sizes = [], []
    exact, possible_rel = 0, 0
    for query, as_of in queries:
        scope = parse_date_scope(query["query"], as_of)
        exact += bool(scope.start and scope.start == scope.end)
        mailbox = query["synthetic_user"]
        all_sizes.append(len(search.candidates(mailbox, as_of)))
        rows = search.candidates(mailbox, as_of, scope)
        possible_rel += sum(documents[row]["doc_id"] in gold[query["query_id"]] for row in rows)
        scoped_sizes.append(len(rows))
    return {
        "documents": len(documents), "queries": len(queries),
        "judgments": sum(map(len, gold.values())),
        "single_rel_queries": sum(len(r) == 1 for r in gold.values()),
        "exact_day_queries": exact, "relevant_in_parsed_scope": possible_rel,
        "median_candidates_visible": statistics.median(all_sizes),
        "median_candidates_scoped": statistics.median(scoped_sizes),
        "max_candidates_scoped": max(scoped_sizes),
        "recall_ceiling_at_k": {
            k: statistics.mean(min(k, len(r)) / len(r) for r in gold.values()) for k in cutoffs
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "mailsearch")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--methods", nargs="+", choices=list(METHODS))
    target.add_argument("--api", help="Base URL of the running MVP, e.g. http://127.0.0.1:8765")
    parser.add_argument("--rewrites", type=Path, help="JSON mapping of replacement phrases for a stress run")
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10, 20, 50])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, help="Write complete per-query results as JSON")
    args = parser.parse_args()
    if args.repeats < 1 or min(args.k) < 1:
        parser.error("--repeats и --k должны быть положительными")
    if args.api and max(args.k) > 100:
        parser.error("В MVP максимальный limit равен 100")
    cutoffs = sorted(set(args.k))
    methods = [] if args.api else list(dict.fromkeys(args.methods or METHODS))

    documents, queries, gold = load_data(args.data_dir)
    if args.rewrites:
        queries = rewrite_queries(queries, args.rewrites)
    documents_by_id = {doc["doc_id"]: doc for doc in documents}
    start = perf_counter()
    print(f"{len(documents)} писем. {f'Сервер {args.api}' if args.api else 'Подготовка индексов'}", flush=True)
    search = Search(documents, methods, args.cache_dir)
    build_seconds = perf_counter() - start
    scope = scope_stats(search, documents, queries, gold, cutoffs)
    print(f"{scope['queries']} запросов, {scope['judgments']} релевантных пар, {scope['single_rel_queries']} одиночных")
    print(f"Фильтр даты сохранил {scope['relevant_in_parsed_scope']}/{scope['judgments']} релевантных пар")
    print(f"Кандидатов на запрос (медиана): {scope['median_candidates_visible']:.0f} → {scope['median_candidates_scoped']:.0f}")
    display_k = 10 if 10 in cutoffs else cutoffs[-1]
    print(f"{'Метод':<21} {'nDCG@'+str(display_k):>9} {'Recall@'+str(display_k):>10} {'MRR@'+str(display_k)+'*':>9} {'p50, мс':>9} {'p99, мс':>9}", flush=True)
    results = {}
    if args.api:
        rankers = {"mvp-api": partial(api_rank, args.api.rstrip("/") + "/api/search")}
    else:
        rankers = {method: partial(local_rank, search, method) for method in methods}
    for method, rank in rankers.items():
        measured = evaluate(rank, queries, documents_by_id, gold, cutoffs, args.repeats)
        results[method] = measured
        metrics = measured["metrics"][display_k]
        speed = measured["latency_ms"]
        mrr = metrics.get("MRR-single")
        mrr_display = f"{mrr:.4f}" if mrr is not None else "—"
        print(f"{method:<21} {metrics['nDCG']:>9.4f} {metrics['Recall']:>10.4f} {mrr_display:>9} {speed['p50']:>9.2f} {speed['p99']:>9.2f}", flush=True)
        if "server_latency_ms" in measured:
            server = measured["server_latency_ms"]
            print(f"  p99 бэкенда по Server-Timing: {server['p99']:.2f} мс (p99 клиента в таблице)")
    print("* MRR рассчитан по запросам с одним релевантным письмом.")

    if args.output:
        report = {
            "platform": platform.platform(), "repeats": args.repeats, "cutoffs": cutoffs,
            "build_seconds": build_seconds, "data": scope, "methods": results, "api_url": args.api,
            "rewrites": str(args.rewrites) if args.rewrites else None,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Отчёт сохранён в {args.output.resolve()}")


if __name__ == "__main__":
    main()
