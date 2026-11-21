"""Compare two saved benchmark.py rankings on a fixed set of mail-search tasks.

Use --tasks for a JSONL file with query_id, family_id, split (dev/blind),
state (positive/verified_empty/unjudged) and task_type (known_item/collection/empty).
Positive known-item tasks also list their acceptable target_ids. Optional source
defines bootstrap strata. Optional synthetic_user and query_date must match the
query. Explicit owner_id, chain_id and all positively judged document IDs are
confined to one split. Optional variant defaults to original; other variants
are secondary analyses. An optional original_query_id links a variant to its
original. --public-templates instead groups the organizers' dev queries for a
descriptive calculation using the keys of tests/fixtures/public_topics.json.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np


K = 10
RESAMPLES = 10_000
SEED = 2026
METRICS = (
    "ndcg_at_10", "known_recall_at_10", "judged_at_10", "returned_at_10_positive",
    "returned_length", "success_at_1", "success_at_5", "mrr_at_10",
    "false_empty_rate", "false_nonempty_rate",
)
RATES = {"false_empty_rate", "false_nonempty_rate", "success_at_1", "success_at_5"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_lines(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    rows = []
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path.name}, строка {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}, строка {line_no}: нужен JSON-объект")
            rows.append(row)
    return rows, sha(raw)


def string(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Поле {field} должно быть непустой строкой")
    return value


def keyed(rows, name):
    records = {}
    for row in rows:
        qid = string(row.get("query_id"), "query_id")
        if qid in records:
            raise ValueError(f"{name}: повтор query_id {qid}")
        records[qid] = row
    return records


def same_ids(actual, expected, name):
    extra, missing = set(actual) - set(expected), set(expected) - set(actual)
    if extra or missing:
        raise ValueError(f"{name}: query_id не совпадают, лишних {len(extra)}, отсутствующих {len(missing)}")


def load_labels(rows, queries):
    labels = {qid: {} for qid in queries}
    for row in rows:
        qid = string(row.get("query_id"), "query_id")
        did = string(row.get("doc_id"), "doc_id")
        grade = row.get("relevance")
        if qid not in labels or did in labels[qid] or type(grade) is not int or grade not in {0, 1, 2}:
            raise ValueError(f"Некорректный или повторный qrel: {qid} / {did}")
        labels[qid][did] = grade
    return labels


@dataclass(frozen=True)
class Task:
    qid: str
    family: str
    split: str
    state: str
    kind: str
    source: str
    mailbox: str
    variant: str
    targets: frozenset[str]


def prepare_tasks(rows, queries, labels):
    metadata = keyed(rows, "tasks")
    same_ids(metadata, queries, "tasks")
    tasks = {}
    families = {}
    memberships = {}

    def check_split(field, identifier, split):
        if memberships.setdefault((field, identifier), split) != split:
            raise ValueError(f"{field} {identifier!r} встречается и в dev, и в blind")

    for qid, row in metadata.items():
        family = string(row.get("family_id"), "family_id")
        split, state, kind = row.get("split"), row.get("state"), row.get("task_type")
        if split not in {"dev", "blind"} or state not in {"positive", "verified_empty", "unjudged"}:
            raise ValueError(f"{qid}: неизвестный split или state")
        if kind not in {"known_item", "collection", "empty"} or (kind == "empty" and state == "positive"):
            raise ValueError(f"{qid}: несовместимый task_type {kind!r} и state {state!r}")
        source = string(row.get("source", "unspecified"), "source")
        if families.setdefault(family, (split, source)) != (split, source):
            raise ValueError(f"Семейство {family} встречается в нескольких split или source")
        for field in ("synthetic_user", "query_date"):
            if field in row and row[field] != queries[qid][field]:
                raise ValueError(f"{qid}: поле {field} в tasks расходится с queries")
        for field in ("owner_id", "chain_id"):
            identifier = row.get(field)
            if identifier is not None:
                check_split(field, string(identifier, field), split)

        positives = {doc for doc, grade in labels[qid].items() if grade > 0}
        if bool(positives) != (state == "positive"):
            raise ValueError(f"{qid}: state {state!r} расходится с положительными qrels")
        target_list = row.get("target_ids", [])
        if not isinstance(target_list, list):
            raise ValueError(f"{qid}: target_ids должен быть списком")
        targets = frozenset(string(did, "target_ids") for did in target_list)
        if len(targets) != len(target_list) or (targets and kind != "known_item"):
            raise ValueError(f"{qid}: некорректные target_ids")
        if state == "positive" and kind == "known_item" and (not targets or not targets <= positives):
            raise ValueError(f"{qid}: допустимые target_ids должны иметь положительные qrels")
        for did in positives | targets:
            check_split("doc_id", did, split)
        origin = row.get("original_query_id")
        default_variant = "original" if origin is None or origin == qid else "variant"
        variant = string(row.get("variant", default_variant), "variant")
        tasks[qid] = Task(qid, family, split, state, kind, source, queries[qid]["synthetic_user"], variant, targets)
    for qid, row in metadata.items():
        origin = row.get("original_query_id")
        if origin is None:
            continue
        task, parent = tasks[qid], tasks.get(origin)
        if parent is None or parent.family != task.family or parent.variant != "original":
            raise ValueError(f"{qid}: original_query_id должен указывать на исходную задачу того же семейства")
        if (task.variant == "original") != (origin == qid) or parent.mailbox != task.mailbox:
            raise ValueError(f"{qid}: несовместимая связь варианта с original_query_id")
    return tasks


def public_tasks(queries, labels, templates):
    if not isinstance(templates, dict) or not templates:
        raise ValueError("Нужен непустой JSON-словарь публичных тематических шаблонов")
    rows = []
    observed = set()
    for qid, q in queries.items():
        matching = [key for key in templates if key in q["query"]]
        if len(matching) != 1:
            raise ValueError(f"{qid}: найдено тематических шаблонов {len(matching)}, требуется один")
        observed.add(matching[0])
        if not labels[qid] or any(grade != 1 for grade in labels[qid].values()):
            raise ValueError("Режим публичных шаблонов предназначен для публичных qrels с положительными единицами")
        positive = [did for did, grade in labels[qid].items() if grade > 0]
        single = len(positive) == 1
        rows.append(dict(query_id=qid, family_id=matching[0], split="dev", source="vk_public_dev",
                         state="positive", task_type="known_item" if single else "collection",
                         target_ids=positive if single else []))
    if observed != set(templates):
        raise ValueError("Некоторые публичные тематические шаблоны не соответствуют ни одному запросу")
    return rows


def read_report(path, queries, labels, queries_sha, labels_sha, label_count):
    raw = path.read_bytes()
    report = json.loads(raw)
    dataset = report["dataset"]
    hashes = dataset["sha256"]
    corpus = hashes.get("documents.jsonl")
    if not isinstance(corpus, str) or not re.fullmatch(r"[0-9a-f]{64}", corpus):
        raise ValueError(f"{path.name}: отсутствует SHA-256 снимка корпуса")
    if hashes.get("queries", hashes.get("queries_dev.jsonl")) != queries_sha:
        raise ValueError(f"{path.name}: SHA-256 файла запросов не совпадает")
    saved_labels = hashes.get("qrels", hashes.get("qrels_dev.jsonl"))
    if saved_labels is None:
        if dataset.get("qrels") != 0 or dataset.get("qrels_file") is not None:
            raise ValueError(f"{path.name}: отсутствует SHA-256 заявленных qrels")
    elif saved_labels != labels_sha or dataset["qrels"] != label_count:
        raise ValueError(f"{path.name}: не совпадает SHA-256 или число строк qrels")
    depth = report["timing"]["top_k"]
    if type(depth) is not int or depth < K or dataset["queries"] != len(queries):
        raise ValueError(f"{path.name}: требуется глубина не меньше 10 и одинаковое число запросов")
    documents = dataset["documents"]
    if type(documents) is not int or documents < 1:
        raise ValueError(f"{path.name}: некорректное число писем в снимке")
    results = report["queries"]
    if not isinstance(results, dict):
        raise ValueError(f"{path.name}: требуется отчёт benchmark.py с ranked_ids по каждому query_id")
    same_ids(results, queries, path.name)
    ranks = {}
    for qid, row in results.items():
        ids = row["ranked_ids"]
        if not isinstance(ids, list) or len(ids) > depth or any(not isinstance(i, str) or not i.strip() for i in ids):
            raise ValueError(f"{path.name}, {qid}: некорректные ranked_ids")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{path.name}, {qid}: повтор документа в выдаче")
        nrel = sum(g > 0 for g in labels[qid].values())
        expected = nrel if labels[qid] and saved_labels is not None else None
        if row.get("relevant") != expected:
            raise ValueError(f"{path.name}, {qid}: число релевантных писем в отчёте расходится с qrels")
        ranks[qid] = ids
    provenance = {"file": path.name, "sha256": sha(raw), "documents_sha256": corpus, "documents": documents,
                  "requested_depth": depth, "server_corpus_check": dataset.get("server_corpus_check", "not_recorded"),
                  "qrels_in_ranking_report": saved_labels is not None}
    return ranks, provenance


def outcomes(ranks, labels, task):
    top = ranks[:K]
    out = {"returned_length": len(ranks)}
    if task.state == "verified_empty":
        out["false_nonempty_rate"] = int(bool(ranks))
    if task.state != "positive":
        return out
    positives = sum(g > 0 for g in labels.values())
    discounts = [math.log2(rank + 2) for rank in range(K)]

    def dcg(grades):
        return sum((2**grade - 1) / weight for grade, weight in zip(grades, discounts))

    out["ndcg_at_10"] = dcg([labels.get(did, 0) for did in top]) / dcg(sorted(labels.values(), reverse=True))
    out["known_recall_at_10"] = sum(labels.get(did, 0) > 0 for did in top) / positives
    out["false_empty_rate"] = int(not ranks)
    out["returned_at_10_positive"] = len(top)
    if top:
        out["judged_documents"] = sum(did in labels for did in top)
        out["judged_at_10"] = out["judged_documents"] / len(top)
    if task.kind == "known_item":
        first = next((i for i, did in enumerate(top, 1) if did in task.targets), None)
        out["success_at_1"] = int(first == 1)
        out["success_at_5"] = int(first is not None and first <= 5)
        out["mrr_at_10"] = 1 / first if first else 0.0
    return out


@dataclass(frozen=True)
class Pair:
    task: Task
    before: dict
    after: dict


def rounded(value):
    return round(float(value), 8) if value is not None else None


def paired_ndcg(pairs, seed, resamples):
    groups = defaultdict(lambda: defaultdict(list))
    diffs = []
    for pair in pairs:
        if pair.task.state == "positive":
            delta = pair.after["ndcg_at_10"] - pair.before["ndcg_at_10"]
            groups[pair.task.source][pair.task.family].append(delta)
            diffs.append(delta)
    n_groups = sum(len(s) for s in groups.values())
    identical = sum(abs(d) <= 1e-12 for d in diffs)
    wins = sum(d > 1e-12 for d in diffs)
    result = {"positive_queries": len(diffs), "positive_families": n_groups,
              "families_by_source": {s: len(g) for s, g in sorted(groups.items())},
              "wins": wins, "ties": identical, "losses": len(diffs) - wins - identical,
              "delta": rounded(statistics.mean(diffs)) if diffs else None, "ci95": None}
    if not any(len(families) > 1 for families in groups.values()):
        result["note"] = "Недостаточно независимых групп внутри источников для интервала."
        return result
    rng = np.random.default_rng(seed)
    delta_sums = np.zeros(resamples)
    counts = np.zeros(resamples)
    for source in sorted(groups):
        families = groups[source]
        values = [families[key] for key in sorted(families)]
        sums = np.array([sum(x) for x in values])
        sizes = np.array([len(x) for x in values])
        n = len(values)
        batch_size = max(1, min(512, 1_000_000 // n))
        for start in range(0, resamples, batch_size):
            stop = min(start + batch_size, resamples)
            draw = rng.integers(n, size=(stop - start, n))
            delta_sums[start:stop] += sums[draw].sum(axis=1)
            counts[start:stop] += sizes[draw].sum(axis=1)
    result["ci95"] = [rounded(x) for x in np.quantile(delta_sums / counts, [0.025, 0.975])]
    if n_groups < 20:
        result["note"] = "Мало независимых групп. Интервал описывает этот набор и чувствителен к его составу."
    return result


def summarize(pairs, seed, resamples):
    metrics = {}
    for name in METRICS:
        before = [r.before[name] for r in pairs if name in r.before]
        after = [r.after[name] for r in pairs if name in r.after]
        a = statistics.mean(before) if before else None
        b = statistics.mean(after) if after else None
        block = {"before": rounded(a), "after": rounded(b),
                 "delta": rounded(b - a) if a is not None and b is not None else None,
                 "n_before": len(before), "n_after": len(after)}
        if name in RATES:
            block.update(events_before=sum(before), events_after=sum(after))
        if name == "judged_at_10":
            block["documents_before"] = sum(p.before.get("judged_documents", 0) for p in pairs)
            block["documents_after"] = sum(p.after.get("judged_documents", 0) for p in pairs)
            block["returned_before"] = sum(p.before.get("returned_at_10_positive", 0) for p in pairs)
            block["returned_after"] = sum(p.after.get("returned_at_10_positive", 0) for p in pairs)
        metrics[name] = block
    states = Counter(p.task.state for p in pairs)
    return {"queries": len(pairs), "families": len({p.task.family for p in pairs}),
            "mailboxes": len({p.task.mailbox for p in pairs}),
            "states": {s: states[s] for s in ("positive", "verified_empty", "unjudged")},
            "task_types": dict(sorted(Counter(p.task.kind for p in pairs).items())),
            "metrics": metrics, "paired_ndcg_at_10": paired_ndcg(pairs, seed, resamples)}


def sections(pairs, seed, resamples):
    splits, sources = defaultdict(list), defaultdict(list)
    for pair in pairs:
        splits[pair.task.split].append(pair)
        sources[pair.task.source].append(pair)
    result = {"all": summarize(pairs, seed, resamples),
              "by_split": {k: summarize(splits[k], seed, resamples) if k in splits else None
                           for k in ("dev", "blind")}}
    if len(sources) > 1:
        result["by_source"] = {k: summarize(v, seed, resamples) for k, v in sorted(sources.items())}
    return result


def compare(before_path, after_path, queries_path, qrels_path, tasks_path=None, *, public_templates=None,
            seed=SEED, resamples=RESAMPLES):
    queries_rows, queries_sha = read_lines(queries_path)
    queries = keyed(queries_rows, "queries")
    if not queries:
        raise ValueError("Нет запросов")
    for q in queries.values():
        string(q.get("query"), "query")
        string(q.get("synthetic_user"), "synthetic_user")
        date.fromisoformat(string(q.get("query_date"), "query_date"))
    qrels_rows, labels_sha = read_lines(qrels_path)
    labels = load_labels(qrels_rows, queries)
    if (tasks_path is None) == (public_templates is None):
        raise ValueError("Укажите один файл метаданных задач или публичных шаблонов")
    if public_templates:
        raw = public_templates.read_bytes()
        tasks_rows = public_tasks(queries, labels, json.loads(raw))
        grouping = {"file": public_templates.name, "sha256": sha(raw), "mode": "public_topic_templates"}
    else:
        tasks_rows, tasks_sha = read_lines(tasks_path)
        grouping = {"file": tasks_path.name, "sha256": tasks_sha, "mode": "provided_families"}
    tasks = prepare_tasks(tasks_rows, queries, labels)
    before, a = read_report(before_path, queries, labels, queries_sha, labels_sha, len(qrels_rows))
    after, b = read_report(after_path, queries, labels, queries_sha, labels_sha, len(qrels_rows))
    for key in ("documents_sha256", "documents", "requested_depth"):
        if a[key] != b[key]:
            raise ValueError(f"Нет парного сравнения: отчёты расходятся по {key}")
    groups = defaultdict(list)
    for qid, task in sorted(tasks.items()):
        groups[task.variant].append(Pair(task, outcomes(before[qid], labels[qid], task),
                                         outcomes(after[qid], labels[qid], task)))
    report = {
        "inputs": {"before": a, "after": b, "queries_sha256": queries_sha, "qrels_sha256": labels_sha,
                   "qrels": len(qrels_rows), "grouping": grouping},
        "protocol": {"depth": K, "delta": "after minus before", "primary": "original formulations only",
                     "coverage": "Judged@10 averages judged/returned among nonempty positive queries. Unjudged documents get zero gain; known recall uses positive qrels.",
                     "length": "returned_length describes all input queries. returned_at_10_positive includes empty positive queries.",
                     "bootstrap": {"metric": "ndcg_at_10", "resamples": resamples, "seed": seed,
                                   "unit": "family_id", "stratify_by": "source",
                                   "estimator": "query-weighted macro mean; paired resampling of intact positive-query clusters within source; percentile interval",
                                   "population": "With few mailbox owners, intervals are conditional on the observed owners."},
                     "candidate_recall": "Unavailable: the reports contain only final ranked results."},
        "primary": sections(groups.pop("original", []), seed, resamples),
        "variants": {k: sections(v, seed, resamples) for k, v in sorted(groups.items())},
    }
    if public_templates:
        report["interpretation"] = ("Описательная диагностика на публичном development. Группы — повторяющиеся темы. "
            "Для совместимости запросы с одной положительной меткой дают условный срез known_item. "
            "Он не устанавливает пользовательское намерение найти одно конкретное письмо. "
            "Публичные qrels содержат только положительные связи.")
    weak = [side for side, inputs in (("before", a), ("after", b)) if inputs["server_corpus_check"] != "sha256"]
    if weak:
        report["snapshot_note"] = ("Входные документы бенчмарка имеют одинаковый SHA-256. Подтверждение "
                                   "этого хеша самим сервером не записано для " + ", ".join(weak) + ".")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "after", "queries", "qrels", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    groups = parser.add_mutually_exclusive_group(required=True)
    groups.add_argument("--tasks", type=Path)
    groups.add_argument("--public-templates", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    try:
        report = compare(args.before, args.after, args.queries, args.qrels, args.tasks,
                         public_templates=args.public_templates, seed=args.seed)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.exit(2, f"Ошибка сравнения: {exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result = report["primary"]["all"]["paired_ndcg_at_10"]
    print(f"{result['positive_queries']} положительных исходных запросов, {result['positive_families']} групп. "
          f"Δ nDCG@10 {result['delta']}, 95% ДИ {result['ci95']}. Отчёт: {args.output}")


if __name__ == "__main__":
    main()
