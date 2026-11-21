"""Run the frozen RuBQ 2.0 passage-retrieval transfer test with the local project venv.

    .venv/bin/python tools/check_rubq.py --prepare-only
    .venv/bin/python tools/check_rubq.py --output results/rubq_transfer.json

The first command only downloads two public files and freezes the adapters and labels.
The second uses two isolated copies of SearchEngine, with the local E5 weights.
"""

import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
import io
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "fade3eae6a4807c35b19cfce20dfa805fc8c0cdf"
URL = f"https://raw.githubusercontent.com/vladislavneon/RuBQ/{UPSTREAM}/RuBQ_2.0"
BLOBS = {
    "RuBQ_2.0_paragraphs.json": "1b27f82170edf9fbae8310d816a983340c9e706c",
    "RuBQ_2.0_test.json": "fd9b04aa5754b7bbc454d875e559bce6d31d323f",
}
COMMITS = ("5fe3f5d3739ac523d0b4088f15eef4b50391dd28", "78979130983382e9ffadcac30aad15fba17c2e9b")
MAILBOX = "rubq"
DAY = "2020-12-15"
SEED = 20261121
BOOTSTRAPS = 10_000


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def write_lines(path, values):
    with path.open("w", encoding="utf-8") as output:
        for row in values:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_lines(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def read_upstream(cache, filename):
    path = cache / "raw" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temp = path.with_suffix(".download")
        subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location", "--retry", "3",
                        "--compressed", f"{URL}/{filename}", "-o", str(temp)], check=True)
        temp.replace(path)
    raw = path.read_bytes()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if blob != BLOBS[filename]:
        raise ValueError(f"{filename} differs from the pinned upstream Git blob. Remove it and retry.")
    return json.loads(raw)


def support_clusters(pages):
    """Connected questions share at least one Wikipedia page providing a labeled answer."""
    by_page = defaultdict(list)
    for qid, answer_pages in pages.items():
        for page in answer_pages:
            by_page[page].append(qid)
    clusters = {}
    for root in sorted(pages):
        if root in clusters:
            continue
        pending = [root]
        while pending:
            qid = pending.pop()
            if qid not in clusters:
                clusters[qid] = root
                for page in pages[qid]:
                    pending.extend(by_page.pop(page, ()))
    return clusters


def freeze(cache):
    marker = cache / "rubq-source-revision"
    if (cache / "documents.jsonl").exists() and (not marker.is_file() or marker.read_text().strip() != UPSTREAM):
        raise ValueError("Choose a dedicated RuBQ cache; an existing corpus already occupies this directory.")
    paragraphs = read_upstream(cache, "RuBQ_2.0_paragraphs.json")
    test = read_upstream(cache, "RuBQ_2.0_test.json")
    marker.write_text(UPSTREAM + "\n", encoding="utf-8")
    pages = {p["uid"]: p["ru_wiki_pageid"] for p in paragraphs}
    if len(pages) != len(paragraphs) or len({q["uid"] for q in test}) != len(test):
        raise ValueError("Upstream question or passage IDs are not unique.")
    docs = ({"doc_id": str(p["uid"]), "synthetic_user": MAILBOX, "doc": p["text"], "doc_dt": None}
            for p in paragraphs)
    write_lines(cache / "documents.jsonl", docs)
    queries, labels, supporting_pages = [], {}, {}
    for q in test:
        hits = sorted(set(q["paragraphs_uids"]["with_answer"]))
        if not hits:
            continue
        qid = str(q["uid"])
        if not q["question_text"].strip() or any(did not in pages or pages[did] is None for did in hits):
            raise ValueError("Invalid test question, confirmed passage, or supporting page.")
        queries.append({"query_id": qid, "query": q["question_text"], "synthetic_user": MAILBOX, "query_date": DAY})
        labels[qid] = [str(did) for did in hits]
        supporting_pages[qid] = {pages[did] for did in hits}
    write_lines(cache / "queries.jsonl", queries)
    write_lines(cache / "qrels.jsonl", ({"query_id": qid, "doc_id": did, "relevance": 1}
                                       for qid, hits in labels.items() for did in hits))
    groups = support_clusters(supporting_pages)
    write_json(cache / "clusters.json", groups)
    return {
        "source_revision": UPSTREAM, "engine_revisions": list(COMMITS),
        "source_url": f"https://github.com/vladislavneon/RuBQ/tree/{UPSTREAM}/RuBQ_2.0",
        "annotation_paper": "https://openreview.net/pdf?id=P5UQFFoQ4PJ",
        "source_license": f"https://github.com/vladislavneon/RuBQ/blob/{UPSTREAM}/LICENSE.txt",
        "raw_sha256": {f: digest(cache / "raw" / f) for f in BLOBS},
        "adapter_sha256": {f: digest(cache / f) for f in ("documents.jsonl", "queries.jsonl", "qrels.jsonl", "clusters.json")},
        "paragraphs": len(paragraphs), "wikipedia_pages": len(set(pages.values())), "official_test_questions": len(test),
        "scored_questions": len(queries), "without_confirmed_passage_excluded": len(test) - len(queries),
        "positive_pairs": sum(map(len, labels.values())), "unique_scored_positive_passages": len(set().union(*labels.values())),
        "support_page_clusters": len(set(groups.values())), "largest_cluster_questions": max(Counter(groups.values()).values()),
        "question_policy": "All original Russian test questions with nonempty paragraphs_uids.with_answer. No subsampling.",
        "qrels_policy": "Binary gain only for author-provided, crowd-verified with_answer IDs. Other IDs receive zero metric gain. No answer strings, SPARQL, related-ID lists or labels are supplied to the ranker.",
        "corpus_policy": "Every official paragraph, in release order. Only original paragraph text, one synthetic mailbox, unknown receipt dates. Fixed query date " + DAY + ".",
        "primary": "Macro nDCG@10, paired candidate minus baseline. Also macro known Recall@10, Hit@10, MRR@10 and counts of empty results for these positive questions.",
        "bootstrap": {"unit": "Transitive components of queries sharing a gold supporting Wikipedia page",
                      "estimator": "Query-weighted paired percentile interval for nDCG@10 delta", "draws": BOOTSTRAPS, "seed": SEED},
    }


def rank(cache, revision):
    tree = cache / "engines" / revision
    tree.mkdir(parents=True, exist_ok=True)
    source = subprocess.check_output(["git", "-C", str(ROOT), "archive", revision, "mailsearch"])
    with tarfile.open(fileobj=io.BytesIO(source)) as archive:
        archive.extractall(tree, filter="data")
    sys.path.insert(0, str(tree))
    import mailsearch
    from mailsearch.retrieval import SearchEngine

    frozen = json.loads((cache / "freeze.json").read_text(encoding="utf-8"))
    for name in ("documents.jsonl", "queries.jsonl"):
        if digest(cache / name) != frozen["adapter_sha256"][name]:
            raise ValueError("Adapted corpus or queries changed after the freeze.")
    documents = read_lines(cache / "documents.jsonl")
    ids = {d["doc_id"] for d in documents}
    engine = SearchEngine(documents, cache / "vectors")
    vectors_sha = digest(Path(engine.dense.vectors.filename))
    ranks = {}
    for q in read_lines(cache / "queries.jsonl"):
        hits = engine.search(q["synthetic_user"], q["query"], date.fromisoformat(q["query_date"]), limit=10)
        result = [did for did, _ in hits]
        if len(result) > 10 or len(set(result)) != len(result) or not ids.issuperset(result):
            raise ValueError("Ranking returned invalid IDs, duplicates or unexpected depth.")
        ranks[q["query_id"]] = result
        if len(ranks) % 250 == 0:
            print(f"{revision[:7]}: {len(ranks)} questions ranked", flush=True)
    encoder = {"model": mailsearch.E5_MODEL, "revision": mailsearch.E5_REVISION,
               "query_model": getattr(mailsearch, "E5_QUERY_GRAPH", "PyTorch fp32")}
    write_json(cache / f"{revision}.json", {"commit": revision, "freeze_sha256": digest(cache / "freeze.json"),
                                            "encoder": encoder, "passage_vectors_sha256": vectors_sha, "ranked_ids": ranks})


def aggregate(cache, frozen):
    from validate_rankings import Pair, Task, outcomes, paired_ndcg

    first, second = [json.loads((cache / f"{rev}.json").read_text(encoding="utf-8")) for rev in COMMITS]
    freeze_sha = digest(cache / "freeze.json")
    if first["passage_vectors_sha256"] != second["passage_vectors_sha256"]:
        raise ValueError("The two experiments used different passage vectors.")
    groups = json.loads((cache / "clusters.json").read_text(encoding="utf-8"))
    for run, revision in zip((first, second), COMMITS):
        if run["commit"] != revision or run["freeze_sha256"] != freeze_sha or set(run["ranked_ids"]) != set(groups):
            raise ValueError("Ranking artifact does not match the frozen protocol.")
    labels = defaultdict(dict)
    for pair in read_lines(cache / "qrels.jsonl"):
        labels[pair["query_id"]][pair["doc_id"]] = pair["relevance"]
    pairs = []
    for qid in sorted(groups):
        task = Task(qid, groups[qid], "blind", "positive", "collection", "rubq", MAILBOX, "original", frozenset())
        scores = []
        for run in (first, second):
            ids = run["ranked_ids"][qid]
            values = outcomes(ids, labels[qid], task)
            first_hit = next((i for i, did in enumerate(ids, 1) if did in labels[qid]), None)
            values["hit_at_10"] = int(first_hit is not None)
            values["mrr_at_10"] = 1 / first_hit if first_hit else 0.0
            scores.append(values)
        pairs.append(Pair(task, *scores))
    metrics = {}
    for name in ("ndcg_at_10", "known_recall_at_10", "hit_at_10", "mrr_at_10"):
        before, after = (mean(getattr(p, side)[name] for p in pairs) for side in ("before", "after"))
        metrics[name] = {"baseline": round(before, 8), "candidate": round(after, 8), "delta": round(after - before, 8)}
    return {"dataset": frozen, "frozen_protocol_sha256": freeze_sha, "entrypoint": "Unmodified SearchEngine.search in separate local processes; shared passage vectors, each release's own query encoder.",
            "encoders": {"baseline": first["encoder"], "candidate": second["encoder"]},
            "shared_passage_vectors_sha256": first["passage_vectors_sha256"], "cutoff": 10, "metrics": metrics,
            "empty_lists_on_positive_queries": {side: sum(not result for result in run["ranked_ids"].values())
                                                for side, run in (("baseline", first), ("candidate", second))},
            "paired_ndcg_at_10": paired_ndcg(pairs, SEED, BOOTSTRAPS),
            "limitations": ["Russian encyclopedic fact questions do not measure personal-mail retrieval, access control, RAG generation, or a VK leaderboard score.",
                            "Only selected candidate passages were manually judged by RuBQ annotators. Additional valid passages in the corpus may have zero measured gain. No verified-empty tasks are available.",
                            "Wikipedia paragraphs lack receipt dates. Each engine retains its normal handling of mail queries, including dates and identifiers. Candidate literal-quotation rules also remain active.",
                            "Pretraining overlap of the shared E5 model with public Wikipedia/RuBQ content was not audited."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache" / "rubq-transfer")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "rubq_transfer.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rank", choices=COMMITS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    cache = args.cache.resolve()
    if args.rank:
        rank(cache, args.rank)
        return
    frozen = freeze(cache)
    write_json(cache / "freeze.json", frozen)
    print(f"Frozen: {frozen['paragraphs']} passages, {frozen['scored_questions']} questions, {frozen['positive_pairs']} positive pairs.", flush=True)
    if args.prepare_only:
        return
    env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="2")
    for rev in COMMITS:
        subprocess.run([sys.executable, __file__, "--cache", str(cache), "--rank", rev], check=True, env=env)
    write_json(args.output, aggregate(cache, frozen))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
