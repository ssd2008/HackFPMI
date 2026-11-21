"""Measure the running search endpoint against a JSONL development benchmark."""

import argparse
from collections import Counter, defaultdict
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

from mailsearch.store import DATA, load_corpus, load_jsonl, verify_manifest


def load_benchmark(data: Path, query_file: Path | None = None, relevance_file: Path | None = None,
                   unjudged: bool = False):
    manifest = verify_manifest(data)
    documents, corpus_sha = load_corpus(data / "documents.jsonl")
    queries_path = query_file or data / "queries_dev.jsonl"
    relevance_path = relevance_file or data / "qrels_dev.jsonl"
    queries = load_jsonl(queries_path)
    have_qrels = not unjudged and (relevance_file is not None or relevance_path.is_file())
    links = load_jsonl(relevance_path) if have_qrels else []
    if not queries:
        raise ValueError("В файле запросов нет строк")
    counts = {"documents": len(documents), "queries": len(queries), "qrels": len(links)}
    check_counts = {"documents": True, "queries": query_file is None,
                    "qrels": have_qrels and relevance_file is None}
    for name, check in check_counts.items():
        if check and name in manifest and counts[name] != manifest[name]:
            raise ValueError(f"Число записей {name} не совпадает с манифестом")
    hashes = {"documents.jsonl": corpus_sha, "queries": hashlib.sha256(queries_path.read_bytes()).hexdigest()}
    if have_qrels:
        hashes["qrels"] = hashlib.sha256(relevance_path.read_bytes()).hexdigest()
    docs = {r["doc_id"]: (r["synthetic_user"], r["doc_dt"]) for r in documents}
    by_query = {r["query_id"]: r for r in queries}
    if len(queries) != len(by_query) or len(documents) != len(docs):
        raise ValueError("Повторяющиеся идентификаторы в исходных данных")
    qrels = defaultdict(dict)
    for link in links:
        qid, did = link["query_id"], link["doc_id"]
        if qid not in by_query or did not in docs:
            raise ValueError(f"В разметке указан неизвестный query_id или doc_id: {qid} / {did}")
        query = by_query[qid]
        box, day = docs[did]
        if (box != query["synthetic_user"] or (day and day > query["query_date"]) or did in qrels[qid]
                or not isinstance(link["relevance"], int) or link["relevance"] < 0):
            raise ValueError(f"Некорректная строка разметки для {qid} / {did}")
        qrels[qid][did] = link["relevance"]
    return docs, queries, qrels, {**counts, "sha256": hashes, "queries_file": queries_path.name,
                                  "qrels_file": relevance_path.name if have_qrels else None}


def check_service_corpus(url: str, docs: dict, expected_hash: str) -> str:
    with urlopen(url.rstrip("/") + "/api/mailboxes", timeout=15) as response:
        metadata = json.load(response)
    if "corpus_sha256" in metadata:
        if metadata["corpus_sha256"] != expected_hash:
            raise ValueError("На сервере другой documents.jsonl. Запустите сервер и бенчмарк с одним --data.")
        return "sha256"
    if Counter(box for box, _ in docs.values()) != {box["id"]: box["count"] for box in metadata["mailboxes"]}:
        raise ValueError("Состав ящиков сервера отличается от корпуса бенчмарка")
    print("Предыдущая версия сервера сообщает только количество писем в ящиках. SHA-256 недоступен.")
    return "mailbox_counts_only"


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
    docs, queries, qrels, dataset = load_benchmark(args.data, args.queries, args.qrels, args.unjudged)
    dataset["server_corpus_check"] = check_service_corpus(args.url, docs, dataset["sha256"]["documents.jsonl"])
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
                labels = qrels[qid]
                nrel = sum(value > 0 for value in labels.values())
                per_query[qid] = {"ranked_ids": result, "relevant": nrel if labels else None,
                                  "scores": {str(k): metrics(result, labels, k) for k in args.k} if nrel else {}}
            elif result != per_query[qid]["ranked_ids"]:
                changed += 1
        print(f"Проход {run_index + 1}/{args.repeats}: {len(order)} запросов", flush=True)
    quality = {}
    for k in args.k:
        values = defaultdict(list)
        for row in per_query.values():
            for metric, value in row["scores"].get(str(k), {}).items():
                values[metric].append(value)
        if values:
            quality[str(k)] = {metric: round(statistics.mean(scores), 6) for metric, scores in values.items()}
    relevant_counts = [row["relevant"] for row in per_query.values() if row["relevant"]]
    report = {
        "dataset": dataset,
        "quality_queries": len(relevant_counts),
        "queries_without_positive_judgments": len(queries) - len(relevant_counts),
        "single_relevant_queries": relevant_counts.count(1),
        "recall_upper_bound": {str(k): round(statistics.mean(min(k, n) / n for n in relevant_counts), 6)
                               for k in args.k if relevant_counts},
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
    print(json.dumps({k: report[k] for k in ("quality_queries", "queries_without_positive_judgments",
                                             "single_relevant_queries", "quality_first_pass", "timing")},
                     ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--queries", type=Path, help="Другой файл запросов к этому корпусу.")
    labels = parser.add_mutually_exclusive_group()
    labels.add_argument("--qrels", type=Path, help="Другой файл оценок релевантности.")
    labels.add_argument("--unjudged", action="store_true", help="Выгрузить выдачу и задержки без метрик качества.")
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
