"""Check original retrieval labels, complete mail filters and split counts against the local corpora."""
import argparse
from collections import defaultdict
from email.utils import parseaddr
import json
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]

def rows(path):
    with path.open(encoding="utf-8") as inp:
        for line in inp:
            if line.strip():
                yield json.loads(line)

def normalize(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().replace("ё", "е").split())

def verify_mail(folder, benchmark):
    docs, positive, judged, _ = benchmark.load_benchmark(folder)
    _, empty, negatives, _ = benchmark.load_benchmark(folder, folder / "queries_empty.jsonl", folder / "qrels_empty.jsonl")
    _, variants, alternate_labels, _ = benchmark.load_benchmark(folder, folder / "queries_variants.jsonl", folder / "qrels_variants.jsonl")
    senders = defaultdict(set)
    subjects = defaultdict(set)
    for doc in rows(folder / "documents.jsonl"):
        first, second, _ = doc["doc"].split("\n", 2)
        mailbox = doc["synthetic_user"]
        senders[mailbox, parseaddr(first.partition(":")[2].strip())[1].lower()].add(doc["doc_id"])
        subjects[mailbox, normalize(second.partition(":")[2])].add(doc["doc_id"])
    metadata = {case["query_id"]: case for case in rows(folder / "cases.jsonl")}
    original = {**judged, **negatives}
    if len(metadata) != len(positive) + len(empty) + len(variants):
        raise ValueError("Missing Enron task metadata")
    exact = 0
    for item in [*positive, *empty]:
        qid = item["query_id"]
        case = metadata[qid]
        gold = {did for did, grade in original.get(qid, {}).items() if grade > 0}
        if bool(gold) != (case['state'] == 'positive') or len(gold) != case['relevant_count']:
            raise ValueError(qid + " has inconsistent labels")
        if case["judgment"] != "exact":
            continue
        box = item["synthetic_user"]
        day = item["query_date"]
        window = case.get("receipt_window")
        if case["family"] == "sender":
            address = item['query'].split('email from ', 1)[1].split(' received', 1)[0]
            ids = senders[box, address]
        else:
            subject = item['query'].split('"', 1)[1].rsplit('"', 1)[0]
            ids = subjects[box, normalize(subject)]
        eligible = {did for did in ids if (not docs[did][1] or docs[did][1] <= day) and
                    (not window or docs[did][1] and window[0] <= docs[did][1] <= window[1])}
        if eligible != gold:
            raise ValueError(qid + " differs from an exhaustive full-mailbox check")
        exact += 1
    for q in variants:
        qid = q['query_id']
        parent = metadata[qid]['original_query_id']
        if alternate_labels.get(qid, {}) != original.get(parent, {}):
            raise ValueError(qid + " lost the original judgments")
    return {'documents': len(docs), 'original_queries': len(positive) + len(empty), 'verified_empty': len(empty), 'rewrites': len(variants), 'exhaustive_exact_tasks': exact}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True, help='Path to the team repository containing benchmark.py')
    parser.add_argument('names', nargs='*', help='Default: all prepared sources')
    args = parser.parse_args()
    sys.path.insert(0, str(args.project.resolve()))
    import benchmark
    base = ROOT / 'external' / 'benchmarks'
    names = args.names or sorted(p.name for p in base.iterdir() if (p / 'manifest.json').is_file())
    for name in names:
        folder = base / name
        if name == 'enron-mail':
            result = verify_mail(folder, benchmark)
        else:
            docs, queries, qrels, info = benchmark.load_benchmark(folder)
            if len(qrels) != len(queries) or any(not any(g > 0 for g in group.values()) for group in qrels.values()):
                raise ValueError(name + ' has unanswerable queries in its judged split')
            result = {k: info[k] for k in ('documents', 'queries', 'qrels')}
        print(name, json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
