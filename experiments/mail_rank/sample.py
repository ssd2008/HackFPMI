"""Build a reproducible Russian retrieval sample from train splits only."""
import gzip
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

DATA = Path("datasets/local")
OUTPUT = DATA / "mail-rank-train.jsonl"
SEED = 20261121
LIMIT = 2000


def miracl(rng):
    root = DATA / "miracl-ru"
    questions = dict(line.rstrip().split("\t", 1) for line in
                     next(root.glob("topics/*/topics/*train.tsv")).open())
    labels = defaultdict(lambda: defaultdict(list))
    for line in next(root.glob("topics/*/qrels/*train.tsv")).read_text().splitlines():
        qid, _, docid, label = line.split()
        labels[qid][int(label)].append(docid)
    groups = []
    for qid in rng.sample(sorted(q for q, rel in labels.items() if rel[1] and rel[0]), LIMIT):
        rel = labels[qid]
        groups.append((qid, rng.choice(rel[1]), rng.sample(rel[0], min(3, len(rel[0])))))
    wanted = {did for _, pos, neg in groups for did in [pos, *neg]}
    passages = {}
    for shard in sorted(root.glob("corpus/*/*.jsonl.gz")):
        with gzip.open(shard, "rb") as src:
            for line in src:
                row = json.loads(line)
                if row["docid"] in wanted:
                    passages[row["docid"]] = row
    for qid, pos, neg in groups:
        yield "miracl", qid, questions[qid], passages[pos], [passages[did] for did in neg]


def tydi(rng):
    # Reservoir sampling retains at most LIMIT query groups in memory.
    pool = []
    seen = 0
    with gzip.open(next((DATA / "mr-tydi-ru").glob("topics/*/train.jsonl.gz")), "rb") as f:
        for line in f:
            row = json.loads(line)
            if not row["positive_passages"] or not row["negative_passages"]:
                continue
            seen += 1
            index = seen - 1 if seen <= LIMIT else rng.randrange(seen)
            if index >= LIMIT:
                continue
            pos = rng.choice(row["positive_passages"])
            negative = {p["docid"]: p for p in row["negative_passages"]}
            for p in row["positive_passages"]:
                negative.pop(p["docid"], None)
            group = ("mr-tydi", row["query_id"], row["query"], pos,
                     rng.sample(list(negative.values()), min(3, len(negative))))
            if index == len(pool):
                pool.append(group)
            else:
                pool[index] = group
    yield from pool


if __name__ == "__main__":
    seen_queries = set()
    tmp = OUTPUT.with_suffix(".tmp")
    with tmp.open("w") as out:
        for sampler in (tydi, miracl):
            for source, qid, query, pos, neg in sampler(random.Random(SEED)):
                normalized = " ".join(query.casefold().split())
                if normalized in seen_queries or not neg:
                    continue
                seen_queries.add(normalized)
                for label, passage in [(1, pos), *((0, p) for p in neg)]:
                    row = dict(source=source, qid=qid, query=query, label=label,
                               docid=passage["docid"], title=passage["title"], text=passage["text"])
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(OUTPUT)
    print(len(seen_queries), "queries", hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
