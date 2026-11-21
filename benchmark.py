"""Measure the running search endpoint against a JSONL development benchmark."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import statistics
import sys
from time import perf_counter
from urllib.error import URLError
from urllib.request import Request, urlopen

from mailsearch.store import DATA, load_jsonl


def load_benchmark(data: Path):
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        with (data / name).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != expected:
                raise ValueError(f"SHA-256 не совпадает у {name}")
    documents = load_jsonl(data / "documents.jsonl")
    queries = load_jsonl(data / "queries_dev.jsonl")
    links = load_jsonl(data / "qrels_dev.jsonl")
    if [len(documents), len(queries), len(links)] != [manifest[k] for k in ("documents", "queries", "qrels")]:
        raise ValueError("Число записей не совпадает с манифестом")
    docs = {r["doc_id"]: (r["synthetic_user"], r["doc_dt"]) for r in documents}
    by_query = {r["query_id"]: r for r in queries}
    if len(queries) != len(by_query) or len(documents) != len(docs):
        raise ValueError("Повторяющиеся идентификаторы в исходных данных")
    qrels = defaultdict(dict)
    for link in links:
        qid, did = link["query_id"], link["doc_id"]
        query = by_query[qid]
        box, day = docs[did]
        if box != query["synthetic_user"] or (day and day > query["query_date"]) or did in qrels[qid]:
            raise ValueError(f"Некорректная строка разметки для {qid} / {did}")
        qrels[qid][did] = link["relevance"]
    if any(not any(x > 0 for x in qrels[qid].values()) for qid in by_query):
        raise ValueError("Есть запрос без положительных оценок")
    return docs, queries, qrels, manifest


def metrics(ranking: list[str], labels: dict[str, int], k: int) -> dict[str, float]:
    top = ranking[:k]
    relevant = {did for did, value in labels.items() if value > 0}
    def dcg(values):
        return sum((2**value - 1) / math.log2(pos + 2) for pos, value in enumerate(values))
    scores = {
        "recall": len(set(top) & relevant) / len(relevant),
        "ndcg": dcg([labels.get(did, 0) for did in top]) / dcg(sorted(labels.values(), reverse=True)[:k]),
    }
    if len(relevant) == 1:
        scores["mrr_single"] = next((1 / pos for pos, did in enumerate(top, 1) if did in relevant), 0.0)
    return scores


def percentiles(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {key: round(ordered[math.ceil(fraction * len(ordered)) - 1], 3)
            for key, fraction in (("p50", .5), ("p95", .95), ("p99", .99), ("max", 1))}


def fetch(url: str, query: dict, limit: int):
    payload = {"mailbox": query["synthetic_user"], "query": query["query"],
               "date": query["query_date"], "limit": limit}
    request = Request(url.rstrip("/") + "/api/search", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    started = perf_counter()
    with urlopen(request, timeout=45) as response:
        content = response.read()
        wall_ms = (perf_counter() - started) * 1000
        server = re.fullmatch(r"search;dur=([\d.]+)", response.headers.get("Server-Timing", ""))
        if server is None:
            raise ValueError("Бэкенд не вернул Server-Timing")
    return [row["id"] for row in json.loads(content)["items"]], float(server[1]), wall_ms


def run(args):
    docs, queries, qrels, manifest = load_benchmark(args.data)
    limit = max(args.k)
    all_server, all_wall, first_server = [], [], []
    per_query = {}
    changed = 0
    def measure(query):
        return fetch(args.url, query, limit)

    for run_index in range(args.repeats):
        order = queries.copy()
        random.Random(2026 + run_index).shuffle(order)
        with ThreadPoolExecutor(max_workers=args.concurrency) as clients:
            responses = list(clients.map(measure, order))
        for query, (result, server_ms, wall_ms) in zip(order, responses, strict=True):
            qid = query["query_id"]
            if len(result) != len(set(result)) or len(result) > limit:
                raise ValueError(f"Повтор или лишнее письмо в выдаче {qid}")
            for did in result:
                box, day = docs[did]
                if box != query["synthetic_user"] or (day and day > query["query_date"]):
                    raise ValueError(f"В выдаче {qid} недоступное письмо {did}")
            all_server.append(server_ms)
            all_wall.append(wall_ms)
            if run_index == 0:
                first_server.append(server_ms)
                per_query[qid] = {"ranked_ids": result, "relevant": len(qrels[qid]),
                                  "scores": {str(k): metrics(result, qrels[qid], k) for k in args.k}}
            elif result != per_query[qid]["ranked_ids"]:
                changed += 1
        print(f"Проход {run_index + 1}/{args.repeats}: {len(order)} запросов", flush=True)
    quality = {}
    for k in args.k:
        values = defaultdict(list)
        for row in per_query.values():
            for metric, value in row["scores"][str(k)].items():
                values[metric].append(value)
        quality[str(k)] = {metric: round(statistics.mean(scores), 6) for metric, scores in values.items()}
    relevant_counts = [sum(x > 0 for x in labels.values()) for labels in qrels.values()]
    report = {
        "dataset": {k: manifest[k] for k in ("documents", "queries", "qrels", "sha256")},
        "single_relevant_queries": relevant_counts.count(1),
        "recall_upper_bound": {str(k): round(statistics.mean(min(k, n) / n for n in relevant_counts), 6)
                               for k in args.k},
        "quality_first_pass": quality,
        "timing": {
            "search_calls": len(all_server), "top_k": limit, "repeats": args.repeats,
            "concurrent_clients": args.concurrency, "seed": 2026, "quantile": "nearest rank",
            "server_ms": percentiles(all_server), "server_first_pass_ms": percentiles(first_server),
            "loopback_http_ms": percentiles(all_wall), "ranking_changes_on_repeats": changed,
            "platform": platform.platform(), "python": platform.python_version(),
        },
        "queries": per_query,
    }
    print(json.dumps({k: report[k] for k in ("single_relevant_queries", "quality_first_pass", "timing")},
                     ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1, help="Число параллельных клиентов.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.k) < 1 or max(args.k) > 100 or min(args.repeats, args.concurrency) < 1:
        parser.error("K должно быть в пределах 1–100, число повторов и клиентов должно быть положительным")
    try:
        run(args)
    except URLError as exc:
        sys.exit(f"Не удалось выполнить поиск на {args.url}. Запустите ./run.sh. Подробности: {exc}")
