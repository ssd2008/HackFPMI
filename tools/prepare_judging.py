"""Локальная подготовка запросов, пула для слепой разметки и согласованных qrels."""

import argparse
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import random
from tempfile import NamedTemporaryFile, TemporaryDirectory


QUERY_KEYS = ("query_id", "synthetic_user", "query", "query_date")
TASK_TYPES = ("known_item", "collection", "empty")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def string_list(values):
    return isinstance(values, list) and all(text(x) for x in values)


def day(value):
    try:
        return isinstance(value, str) and date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def lines(source):
    return [json.loads(line) for line in source.splitlines() if line.strip()]


def jsonl(rows):
    return b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8") for row in rows)


def save_new(path, contents):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    staged = NamedTemporaryFile(dir=path.parent, delete=False)
    temp = Path(staged.name)
    try:
        with staged:
            staged.write(contents)
        # The complete, owner-readable file becomes visible in one operation.
        os.link(temp, path)
    except FileExistsError:
        raise ValueError("Выходной файл уже существует. Укажите новый путь.") from None
    finally:
        temp.unlink(missing_ok=True)


def create_owner_forms(dev_data, blind_data, destination):
    require(not destination.exists() and not destination.is_symlink(),
            "Каталог бланков уже существует. Для нового пилота укажите новый путь.")
    owners = []
    for owner, split, data_dir in (("owner-01", "dev", dev_data), ("owner-02", "blind", blind_data)):
        digest = hashlib.sha256()
        boxes = set()
        with (data_dir / "documents.jsonl").open("rb") as source:
            for line in source:
                digest.update(line)
                if line.strip():
                    box = json.loads(line)["synthetic_user"]
                    require(text(box), "В корпусе отсутствует идентификатор ящика.")
                    boxes.add(box)
        require(boxes, "В одном из двух снимков нет писем.")
        owners.append({"owner_id": owner, "split": split, "documents_sha256": digest.hexdigest(),
                       "mailboxes": sorted(boxes)})
    require(owners[0]["documents_sha256"] != owners[1]["documents_sha256"],
            "Для dev и blind укажите снимки двух разных владельцев.")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as staging:
        package = Path(staging) / "forms"
        package.mkdir(mode=0o700)
        save_new(package / "owners.json", (json.dumps(owners, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        for owner in owners:
            split = owner["split"]
            forms = [
                {"query_id": f"{split}-{i:02d}", "owner_id": owner["owner_id"], "split": split,
                 "family_id": "", "task_type": "", "synthetic_user": "", "query_date": "",
                 "query": "", "goal": "", "success_criterion": "", "known_doc_ids": [],
                 "manual_candidate_ids": [], "state": "", "empty_verification": ""}
                for i in range(1, 7)
            ]
            save_new(package / split / "tasks.jsonl", jsonl(forms))
        # A complete nonempty directory cannot replace another nonempty directory.
        package.rename(destination)


def load_context(args):
    raw = (args.data / "documents.jsonl").read_bytes()
    snapshot = hashlib.sha256(raw).hexdigest()
    rows = lines(raw)
    documents = {r["doc_id"]: r for r in rows}
    require(len(documents) == len(rows), "В корпусе повторяется doc_id.")
    require(all(text(r["doc_id"]) and text(r["synthetic_user"]) and isinstance(r["doc"], str)
                and (r["doc_dt"] is None or day(r["doc_dt"])) for r in rows),
            "В снимке неверны ID, текст письма или дата получения.")
    owners = json.loads(args.owners.read_bytes())
    names = set()
    assignments = set()
    for entry in owners:
        require(text(entry["owner_id"]) and entry["owner_id"] not in names and entry["split"] in ("dev", "blind"),
                "В реестре повторён владелец или не задано разбиение.")
        names.add(entry["owner_id"])
        require(string_list(entry["mailboxes"]) and entry["mailboxes"], "В реестре не заданы ящики владельца.")
        for box in entry["mailboxes"]:
            pair = (entry["documents_sha256"], box)
            require(pair not in assignments, "Один ящик снимка указан у нескольких владельцев.")
            assignments.add(pair)
    owner = next((entry for entry in owners if entry["owner_id"] == args.owner), None)
    require(owner is not None and owner["documents_sha256"] == snapshot, "Снимок или владелец не соответствует реестру.")
    permitted = set(owner["mailboxes"])
    require(permitted <= {r["synthetic_user"] for r in rows}, "В снимке отсутствует один из ящиков владельца.")
    tasks = lines(args.tasks.read_bytes())
    require(tasks, "Файл задач пуст.")
    ids = set()
    for t in tasks:
        require(t.get("owner_id") == args.owner and t.get("split") == owner["split"],
                "В одном пакете разрешены задачи одного владельца и его зафиксированной части.")
        require(all(text(t.get(k)) for k in (*QUERY_KEYS, "goal", "success_criterion")),
                "Заполните текст запроса, ящик, дату, цель и критерий успеха до подготовки выдач.")
        if "task_type" in t:
            require(t["task_type"] in TASK_TYPES, "Укажите намерение: known_item, collection или empty.")
        require(t["query_id"] not in ids and t["synthetic_user"] in permitted and day(t["query_date"]),
                "Повторён ID задачи, неверно указан ящик владельца или календарная дата.")
        ids.add(t["query_id"])
        for field in ("known_doc_ids", "manual_candidate_ids"):
            values = t.get(field, [])
            require(string_list(values), "Известные цели и дополнительные кандидаты задаются списками ID.")
            for doc_id in values:
                check_visible(documents, t, doc_id)
    return tasks, documents, snapshot


def check_visible(documents, task, doc_id):
    row = documents.get(doc_id)
    require(row is not None and row["synthetic_user"] == task["synthetic_user"]
            and (row["doc_dt"] is None or row["doc_dt"] <= task["query_date"]),
            "В пуле неизвестное, чужое или будущее письмо. Проверьте исходные данные.")


def query_rows(tasks):
    return [{k: t[k] for k in QUERY_KEYS} for t in tasks]


def query_hash(tasks, path):
    expected = query_rows(tasks)
    if path is None:
        return hashlib.sha256(jsonl(expected)).hexdigest()
    raw = path.read_bytes()
    given = query_rows(lines(raw))
    require(len(expected) == len(given) and {tuple(r.values()) for r in given} == {tuple(r.values()) for r in expected},
            "Файл запросов предсказаний не совпадает с записанными задачами.")
    return hashlib.sha256(raw).hexdigest()


def make_pool(args, tasks, documents, snapshot):
    require(len(args.predictions) >= 2 and len({p.resolve() for p in args.predictions}) == len(args.predictions),
            "Для пула нужны минимум два отдельных файла предсказаний.")
    require(1 <= args.depth <= 100, "Глубина пула должна быть от 1 до 100.")
    queries_sha = query_hash(tasks, args.queries_file)
    asked = {t["query_id"] for t in tasks}
    candidates = {t["query_id"]: set(t.get("known_doc_ids", [])) | set(t.get("manual_candidate_ids", [])) for t in tasks}
    for path in args.predictions:
        report = json.loads(path.read_bytes())
        sha = report["dataset"]["sha256"]
        require(sha.get("documents.jsonl") == snapshot and sha.get("queries") == queries_sha,
                "Снимок писем или файл запросов в предсказаниях отличается.")
        require(set(report["queries"]) == asked and report["timing"]["top_k"] >= args.depth,
                "В предсказаниях другой набор задач или запрошено меньше кандидатов, чем глубина пула.")
        for task in tasks:
            qid = task["query_id"]
            ranking = report["queries"][qid]["ranked_ids"]
            require(string_list(ranking), "ranked_ids должен быть списком идентификаторов.")
            for did in ranking[:args.depth]:
                check_visible(documents, task, did)
                candidates[qid].add(did)
    rng = random.SystemRandom() if args.seed is None else random.Random(args.seed)
    items = []
    for t in tasks:
        shuffled = sorted(candidates[t["query_id"]])
        rng.shuffle(shuffled)
        items.append({
            **{k: t[k] for k in (*QUERY_KEYS, "goal", "success_criterion")},
            "candidates": [{"doc_id": did, "doc_dt": documents[did]["doc_dt"],
                            "doc": documents[did]["doc"], "relevance": None} for did in shuffled]
        })
    return {"owner_id": args.owner, "split": tasks[0]["split"], "documents_sha256": snapshot, "tasks": items}


def export_qrels(args, tasks, documents, snapshot):
    blank = json.loads(args.pool.read_bytes())
    judged = json.loads(args.consensus.read_bytes())
    require(blank.get("owner_id") == args.owner and blank.get("split") == tasks[0]["split"]
            and blank.get("documents_sha256") == snapshot, "Эталонный пул относится к другой части или снимку.")
    for k in ("owner_id", "split", "documents_sha256"):
        require(blank[k] == judged.get(k), "У согласованной разметки изменились параметры пула.")
    source = {t["query_id"]: t for t in tasks}
    require(len(blank["tasks"]) == len(source) and {t["query_id"] for t in blank["tasks"]} == set(source),
            "Список задач пула отличается от исходного.")
    reviews = {r["query_id"]: r for r in judged["tasks"]}
    require(len(judged["tasks"]) == len(source) and set(reviews) == set(source), "Не все задачи совпадают с копией пула.")
    output, unjudged = [], 0
    for original in blank["tasks"]:
        qid = original["query_id"]
        task, review = source[qid], reviews[qid]
        for field in (*QUERY_KEYS, "goal", "success_criterion"):
            require(task[field] == original[field] == review[field], "В процессе разметки изменили условия задачи.")
        graded = {c["doc_id"]: c for c in review["candidates"]}
        originals = original["candidates"]
        ids = {c["doc_id"] for c in originals}
        require(len(originals) == len(ids) == len(graded) == len(review["candidates"]) and set(graded) == ids,
                "Список кандидатов разметки отличается от исходного пула.")
        manual = set(task.get("known_doc_ids", [])) | set(task.get("manual_candidate_ids", []))
        require(manual <= ids,
                "Пропущена известная цель или добавленный владельцем кандидат.")
        for c in originals:
            did = c["doc_id"]
            check_visible(documents, task, did)
            row = graded[did]
            require(c["relevance"] is None and row["doc"] == c["doc"] == documents[did]["doc"]
                    and row["doc_dt"] == c["doc_dt"] == documents[did]["doc_dt"],
                    "Проверьте исходный неразмеченный пул и тексты показанных писем.")
            grade = row["relevance"]
            if grade is None:
                unjudged += 1
                continue
            require(type(grade) is int and grade in (0, 1, 2), "Допустимы только 0, 1, 2 и null.")
            output.append({"query_id": qid, "doc_id": did, "relevance": grade})
    return output, unjudged


def comparison_metadata(tasks, documents, qrels_path):
    indexed = {task["query_id"]: task for task in tasks}
    labels = {qid: {} for qid in indexed}
    for row in lines(qrels_path.read_bytes()):
        qid, did, grade = row["query_id"], row["doc_id"], row["relevance"]
        require(qid in indexed and text(did) and did not in labels[qid]
                and type(grade) is int and grade in (0, 1, 2), "В qrels неизвестная задача, повтор или неверная оценка.")
        check_visible(documents, indexed[qid], did)
        labels[qid][did] = grade
    metadata = []
    for task in tasks:
        qid, kind, state = task["query_id"], task.get("task_type"), task.get("state")
        require(kind in TASK_TYPES and state in ("positive", "verified_empty", "unjudged"),
                "Для сравнения явно заполните task_type и итоговый state каждой задачи.")
        positives = {did for did, grade in labels[qid].items() if grade > 0}
        require(bool(positives) == (state == "positive") and not (kind == "empty" and positives),
                "Состояние задачи или её тип противоречат положительным qrels.")
        if state == "verified_empty":
            require(text(task.get("empty_verification")) and not task.get("known_doc_ids"),
                    "Для verified_empty нужна запись независимой проверки отсутствия цели.")
        targets = []
        if kind == "known_item" and state == "positive":
            targets = task.get("known_doc_ids", [])
            require(bool(targets) and len(targets) == len(set(targets)) and set(targets) <= positives,
                    "Для known_item укажите подтверждённые целевые ID с положительными оценками 1 или 2.")
        data = {"query_id": qid, "family_id": task.get("family_id") or qid, "split": task["split"],
                "source": task["owner_id"], "owner_id": task["owner_id"], "synthetic_user": task["synthetic_user"],
                "query_date": task["query_date"], "task_type": kind, "state": state, "target_ids": targets}
        for key in ("variant", "original_query_id", "chain_id"):
            if task.get(key):
                data[key] = task[key]
        metadata.append(data)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Создать 12 пустых бланков для двух владельцев.")
    init.add_argument("--dev-data", type=Path, required=True, help="Снимок первого владельца с documents.jsonl.")
    init.add_argument("--blind-data", type=Path, required=True, help="Снимок второго владельца с documents.jsonl.")
    init.add_argument("--output", type=Path, required=True, help="Новый локальный каталог с бланками и реестром.")
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--owners", type=Path, required=True, help="Локальный реестр владельцев и снимков (массив JSON).")
    shared.add_argument("--owner", required=True, help="Идентификатор владельца пакета.")
    shared.add_argument("--data", type=Path, required=True, help="Каталог с documents.jsonl.")
    shared.add_argument("--tasks", type=Path, required=True, help="JSONL с задачами, записанными владельцем.")
    shared.add_argument("--output", type=Path, required=True, help="Новый локальный файл результата.")
    commands.add_parser("queries", parents=[shared], help="Сформировать четыре поля запроса для benchmark.py.")
    pool = commands.add_parser("pool", parents=[shared], help="Сформировать перемешанный пул без оценок.")
    pool.add_argument("--predictions", type=Path, nargs="+", required=True, help="Минимум два JSON-отчёта benchmark.py.")
    pool.add_argument("--queries-file", type=Path, help="Исходный файл запросов, использованный при построении выдач.")
    pool.add_argument("--depth", type=int, default=10)
    pool.add_argument("--seed", type=int, help="Зафиксировать порядок для воспроизведения.")
    export = commands.add_parser("qrels", parents=[shared], help="Выгрузить согласованные людьми оценки.")
    export.add_argument("--pool", type=Path, required=True, help="Первоначальный пул с пустыми оценками.")
    export.add_argument("--consensus", type=Path, required=True, help="Копия пула с итоговыми ручными оценками.")
    metadata = commands.add_parser("metadata", parents=[shared], help="Записать метаданные для парного сравнения.")
    metadata.add_argument("--qrels", type=Path, required=True, help="Итоговые ручные оценки из команды qrels.")
    args = parser.parse_args()
    try:
        if args.command == "init":
            create_owner_forms(args.dev_data, args.blind_data, args.output)
            print("Созданы 12 пустых задач, по шесть для dev и blind.")
            return
        tasks, docs, snap = load_context(args)
        if args.command == "queries":
            contents = jsonl(query_rows(tasks))
            summary = f"Подготовлено запросов: {len(tasks)}."
        elif args.command == "pool":
            result = make_pool(args, tasks, docs, snap)
            contents = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            summary = f"Задач: {len(tasks)}. Пар для разметки: {sum(len(t['candidates']) for t in result['tasks'])}."
        elif args.command == "metadata":
            contents = jsonl(comparison_metadata(tasks, docs, args.qrels))
            summary = f"К парному сравнению подготовлено задач: {len(tasks)}."
        else:
            rows, skipped = export_qrels(args, tasks, docs, snap)
            contents = jsonl(rows)
            summary = f"Экспортировано оценок: {len(rows)}. Остаются без оценки: {skipped}."
        save_new(args.output, contents)
        print(summary)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Ошибка подготовки: {exc}\n")


if __name__ == "__main__":
    main()
