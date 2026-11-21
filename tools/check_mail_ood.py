"""Evaluate mail OOD suites on paired /api/search rankings.

Supply --pool with the distributed validation/mail-ood-v1 folder (including
both documents.jsonl snapshots) and --runs with a local private directory.
--run executes the publisher's run_pool.py against the four specified URLs.
Without it, existing run_pool.py reports in <runs>/<suite>/{before,after}/
are verified and reused. Panels use the same HTTP rankings as the full run.
Only aggregate results and SHA-256 fingerprints are written under --output.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import time
from urllib.parse import urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import metrics as cutoff_metrics
from tools.validate_rankings import keyed, load_labels, outcomes, prepare_tasks, read_report, same_ids

SOURCE = "9aa8aa1173ce3257af22fedf59cc8f3948044653"
BEFORE = "5fe3f5d3739ac523d0b4088f15eef4b50391dd28"
AFTER = "b37c309ff3d9e83f80a726325a3b9b0ee56c33c1"
SUITES = ("vk_dev", "personal_dev")
# The additional splits are local releases. Pin the manifests and panel lists,
# which do not contain recoverable message or query text in this repository.
LOCAL_FINGERPRINTS = {
    "vk_holdout": ("7636c274c33ab1c58e2add793bc238f42bd24a045a077aab824770fae6e9207b",
                   "150802da79054d6ce1607224615eea87f2da453fa6490034373ae7d6ba122157"),
    "personal_holdout": ("32f83d8b5f3a10c1cadc97619da5d011f2c80b98f4997323cbdf62232bd6adde",
                         "297a9cd9eb9ff6a6864c949a0321685a96721e2799615ccfd5a046f17a06c5ba"),
}
SIDES = ("before", "after")
SEED = 2026
RESAMPLES = 10_000
METRICS = ("ndcg_at_1", "ndcg_at_5", "ndcg_at_10", "known_recall_at_1",
           "known_recall_at_5", "known_recall_at_10", "recall_ceiling_at_10",
           "success_at_1", "success_at_5", "mrr_at_10", "singleton_mrr_at_10", "false_empty_rate",
           "false_nonempty_rate", "returned_length", "judged_at_10")
CI_METRICS = ("ndcg_at_10", "known_recall_at_10", "success_at_1", "mrr_at_10", "singleton_mrr_at_10",
              "false_empty_rate", "false_nonempty_rate")
SCOPE = ("returned", "unknown_id", "foreign_mailbox", "future",
         "requested_window_returned", "outside_requested_window",
         "undated_in_requested_window", "queries_outside_requested_window")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def lines(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def confirm(ok, why):
    if not ok:
        raise ValueError(why)


def pinned(path, revision, gitpath):
    frozen = subprocess.check_output(["git", "show", f"{revision}:{gitpath}"], cwd=ROOT)
    confirm(path.read_bytes() == frozen, f"File differs from the pinned revision: {gitpath}")


def commit_id(revision):
    return subprocess.check_output(["git", "rev-parse", "--verify", "--end-of-options",
                                    revision + "^{commit}"], cwd=ROOT, text=True).strip()


def check_inputs(pool, name):
    ds = pool / "suite" / name
    manifest = ds / "manifest.json"
    panel = pool / "panels" / (name + ".jsonl")
    if name in LOCAL_FINGERPRINTS:
        confirm((digest(manifest), digest(panel)) == LOCAL_FINGERPRINTS[name], f"Local release changed: {name}")
    else:
        for file, rel in ((manifest, f"suite/{name}/manifest.json"), (panel, f"panels/{name}.jsonl")):
            pinned(file, SOURCE, "validation/mail-ood-v1/" + rel)
    declared = json.loads(manifest.read_bytes())
    for file, expected in declared["sha256"].items():
        confirm(digest(ds / file) == expected, f"{name}: checksum mismatch for {file}")
    return declared


def run_http(pool, runs, urls, workers):
    program = pool / "method/run_pool.py"
    pinned(program, SOURCE, "validation/mail-ood-v1/method/run_pool.py")
    pinned(ROOT / "benchmark.py", AFTER, "benchmark.py")

    def evaluate(job):
        name, side, url, panel = job
        out = (runs / "panels" if panel else runs) / name / side
        out.mkdir(parents=True, exist_ok=True)
        confirm(not (out / "rankings.json").exists(), "Choose a fresh --runs directory for a new run")
        selector = ["--panel", str(pool / "panels" / (name + ".jsonl"))] if panel else []
        command = [sys.executable, str(program), "--project", str(ROOT),
                   "--suite", str(pool / "suite" / name), "--variants", "--workers",
                   str(workers), "--url", url, "--output", str(out), *selector]
        path = out / "http.log"
        for attempt in range(3):
            with path.open("wb") as log:
                run = subprocess.run(command, cwd=ROOT, stdout=log, stderr=log)
            if run.returncode == 0:
                return
            # The publisher opens a new loopback connection per query. A large
            # suite can temporarily exhaust the client's ephemeral TCP ports.
            failure = path.read_text(errors="replace").casefold()
            if attempt == 2 or not any(why in failure for why in ("assign requested address", "address not available")):
                run.check_returncode()
            time.sleep(20 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=4) as group:
        for panel in (False, True):
            list(group.map(evaluate, ((name, side, urls[name][side], panel) for name in urls for side in SIDES)))


def archive_evidence(runs, url, revision, documents_sha):
    """Verify the optional local launch record, exact extracted source files and corpus."""
    port = urlparse(url).port
    work = runs / f"server-{port}"
    note = runs / f"server-{port}.json"
    if not note.exists():
        return None
    record = json.loads(note.read_bytes())
    archive = subprocess.check_output(["git", "archive", revision, "mailsearch", "static"], cwd=ROOT)
    archive_sha = hashlib.sha256(archive).hexdigest()
    confirm(record["revision"] == revision and record["source_archive_sha256"] == archive_sha
            and record["corpus_sha256"] == documents_sha and record["port"] == port
            and digest(work / "data/documents.jsonl") == documents_sha, "Launch record mismatch")
    with tarfile.open(fileobj=io.BytesIO(archive)) as frozen:
        for member in frozen:
            if member.isfile():
                confirm((work / member.name).read_bytes() == frozen.extractfile(member).read(), "Server source changed")
    return {"revision": revision, "source_archive_sha256": archive_sha, "corpus_sha256": documents_sha}


@dataclass
class Case:
    meta: dict
    values: tuple[dict, dict]
    scope: tuple[dict, dict]


def measure(ranks, task, labels, meta, query, documents):
    values = outcomes(ranks, labels, task)
    if task.state == "positive":
        for k in (1, 5):
            cutoff = cutoff_metrics(ranks, labels, k)
            values[f"ndcg_at_{k}"] = cutoff["ndcg"]
            values[f"known_recall_at_{k}"] = cutoff["recall"]
        n = sum(g > 0 for g in labels.values())
        values["recall_ceiling_at_10"] = min(n, 10) / n
        if n == 1:
            values["singleton_mrr_at_10"] = cutoff_metrics(ranks, labels, 10)["mrr_single"]
    span = meta.get("receipt_window")
    scope = Counter(returned=len(ranks), requested_window_returned=len(ranks) if span else 0)
    for did in ranks:
        if did not in documents:
            scope["unknown_id"] += 1
            continue
        box, day = documents[did]
        scope["foreign_mailbox"] += box != query["synthetic_user"]
        scope["future"] += bool(day and day > query["query_date"])
        if span:
            scope["undated_in_requested_window"] += day is None
            scope["outside_requested_window"] += day is None or not span[0] <= day <= span[1]
    scope["queries_outside_requested_window"] = int(scope["outside_requested_window"] > 0)
    return values, scope


def load_cases(pool, runs, name, manifest, revisions):
    ds = pool / "suite" / name
    positive, empty, variants = (lines(ds / f"queries_{suffix}.jsonl") for suffix in ("dev", "empty", "variants"))
    ordered = positive + empty + variants
    queries = keyed(ordered, "queries")
    links = [l for suffix in ("dev", "empty", "variants") for l in lines(ds / f"qrels_{suffix}.jsonl")]
    labels = load_labels(links, queries)
    records = lines(ds / "cases.jsonl")
    metadata = keyed(records, "cases")
    tasks = prepare_tasks(records, queries, labels)
    documents = {d["doc_id"]: (d["synthetic_user"], d["doc_dt"]) for d in lines(ds / "documents.jsonl")}
    confirm(len(documents) == manifest["documents"], "Duplicate corpus IDs")
    for qid, qrels in labels.items():
        m, q = metadata[qid], queries[qid]
        confirm(m["judgment"] in {"exact", "assistant_reviewed", "semantic_silver"}, "Unknown judgment")
        span = m.get("receipt_window")
        if span:
            confirm(len(span) == 2 and span[0] <= span[1] <= q["query_date"], "Invalid receipt window")
        for did, grade in qrels.items():
            box, day = documents[did]
            confirm(box == q["synthetic_user"] and (not day or day <= q["query_date"]), "Invisible qrel")
            if span and grade:
                confirm(day and span[0] <= day <= span[1], "Positive qrel outside receipt window")
        origin = m.get("original_query_id")
        if origin:
            confirm(labels[origin] == qrels and m["task_id"] == metadata[origin]["task_id"]
                    and m["family_id"] == metadata[origin]["family_id"], "Variant/primary mismatch")
    panel_rows = lines(pool / "panels" / f"{name}.jsonl")
    panel = keyed(panel_rows, "panel")
    confirm(len(panel) == 250 and Counter(r["task_type"] for r in panel_rows)
            == {"known_item": 100, "collection": 100, "empty": 50}, "Panel composition changed")
    for qid, row in panel.items():
        confirm(qid in metadata and not metadata[qid].get("variant")
                and all(row[k] == metadata[qid][k] for k in ("family", "judgment", "task_type")),
                "Panel content is inconsistent with case metadata")
    provenance, ranks, official = {}, [], []
    for side in SIDES:
        out = runs / name / side
        confirm(lines(out / "queries.jsonl") == ordered and lines(out / "qrels.jsonl") == links
                and lines(out / "tasks.jsonl") == [metadata[q["query_id"]] for q in ordered],
                "HTTP workload differs from the pinned suite")
        rank, origin = read_report(out / "rankings.json", queries, labels, digest(out / "queries.jsonl"),
                                   digest(out / "qrels.jsonl"), len(links))
        timings = json.loads((out / "rankings.json").read_bytes())["timing"]
        confirm(origin["documents_sha256"] == manifest["sha256"]["documents.jsonl"]
                and origin["documents"] == len(documents) and origin["requested_depth"] == 10
                and timings["repeats"] == 1 and timings["search_calls"] == len(queries), "Ranking setup mismatch")
        confirm(origin["server_corpus_check"] == ("mailbox_counts_only" if revisions[side] == BEFORE else "sha256"),
                "Unexpected HTTP corpus check")
        provenance[side] = {"ranking_sha256": origin["sha256"], "server_snapshot_check": origin["server_corpus_check"],
                            "requests": timings["search_calls"], "requested_k": 10,
                            "concurrency": timings["concurrent_clients"]}
        ranks.append(rank)
        official.append(json.loads((out / "summary.json").read_bytes())["primary"])
        repeat = runs / "panels" / name / side
        if (repeat / "rankings.json").exists():
            members = {qid for qid, m in metadata.items() if qid in panel or m.get("original_query_id") in panel}
            subset_queries = {qid: queries[qid] for qid in members}
            subset_labels = {qid: labels[qid] for qid in members}
            confirm(lines(repeat / "queries.jsonl") == [q for q in ordered if q["query_id"] in members]
                    and lines(repeat / "qrels.jsonl") == [l for l in links if l["query_id"] in members],
                    "Independent HTTP panel workloads changed")
            reranked, proof = read_report(repeat / "rankings.json", subset_queries, subset_labels,
                                          digest(repeat / "queries.jsonl"), digest(repeat / "qrels.jsonl"),
                                          sum(map(len, subset_labels.values())))
            confirm(proof["documents_sha256"] == origin["documents_sha256"] and proof["requested_depth"] == 10,
                    "Independent panel repeat used a different snapshot or depth")
            confirm(all(rank[qid] == ids for qid, ids in reranked.items()), "Panel rankings changed across runs")
            provenance[side]["independent_panel_requests"] = len(reranked)
            provenance[side]["panel_rankings_sha256"] = proof["sha256"]
            provenance[side]["panel_rankings_different_from_full_run"] = 0
    by_q = {}
    for qid, query in queries.items():
        first, second = (measure(rank[qid], tasks[qid], labels[qid], metadata[qid], query, documents) for rank in ranks)
        by_q[qid] = Case(metadata[qid], (first[0], second[0]), (first[1], second[1]))
    same_ids(by_q, queries, "paired responses")
    return by_q, panel, provenance, official


def round8(value):
    return round(float(value), 8)


def paired_ci(rows, metric, field):
    """Resample eligible families. Means retain query weights within the sampled families."""
    by = defaultdict(list)
    optional_error = metric in {"false_nonempty_rate", "false_empty_rate"}
    for row in rows:
        before, after = row.values
        if metric in before and metric in after:
            by[row.meta[field]].append(after[metric] - before[metric])
    gaps = [x for values in by.values() for x in values]
    larger = sum(x > 1e-12 for x in gaps)
    smaller = sum(x < -1e-12 for x in gaps)
    result = {"queries": len(gaps), "groups": len(by), "wins": smaller if optional_error else larger,
              "ties": len(gaps) - smaller - larger, "losses": larger if optional_error else smaller,
              "delta": round8(statistics.mean(gaps)) if gaps else None, "ci95": None}
    if len(by) < 2:
        return result
    arrays = [by[key] for key in sorted(by)]
    totals, counts = np.array([sum(a) for a in arrays]), np.array([len(a) for a in arrays])
    rng = np.random.default_rng(SEED)
    boot = np.empty(RESAMPLES)
    batch = min(512, max(1, 700_000 // len(by)))
    for start in range(0, RESAMPLES, batch):
        end = min(RESAMPLES, start + batch)
        draw = rng.integers(len(by), size=(end-start, len(by)))
        boot[start:end] = totals[draw].sum(axis=1) / counts[draw].sum(axis=1)
    result["ci95"] = [round8(x) for x in np.quantile(boot, [.025, .975])]
    return result


def summarize(rows, intervals=()):
    metric = {}
    for key in METRICS:
        values = [[row.values[side][key] for row in rows if key in row.values[side]] for side in range(2)]
        means = [statistics.mean(v) if v else None for v in values]
        if not any(values):
            continue
        block = {"before": round8(means[0]) if means[0] is not None else None,
                 "after": round8(means[1]) if means[1] is not None else None,
                 "delta": round8(means[1] - means[0]) if None not in means else None,
                 "n": [len(v) for v in values]}
        if key in {"false_empty_rate", "false_nonempty_rate", "success_at_1", "success_at_5"}:
            block["events"] = [sum(v) for v in values]
        metric[key] = block
    block = {"queries": len(rows), "positive": sum(row.meta["state"] == "positive" for row in rows),
             "verified_empty": sum(row.meta["state"] == "verified_empty" for row in rows),
             "judgments": dict(sorted(Counter(r.meta["judgment"] for r in rows).items())),
             "metrics": metric,
             "scope": {"queries_with_explicit_window": sum(bool(r.meta.get("receipt_window")) for r in rows),
                       **{k: [sum(row.scope[side][k] for row in rows) for side in range(2)] for k in SCOPE}}}
    if intervals:
        block["paired_by_family_id"] = {key: paired_ci(rows, key, "family_id") for key in intervals if key in metric}
    return block


def breakdown(rows, key, intervals=()):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return {k: summarize(group, intervals) for k, group in sorted(groups.items())}


def results(pool, runs, name, manifest, revisions):
    cases, panel, provenance, official = load_cases(pool, runs, name, manifest, revisions)
    primary = [c for c in cases.values() if not c.meta.get("variant")]
    variants = [c for c in cases.values() if c.meta.get("variant")]
    confirm(len(primary) == manifest["primary_tasks"] and len(variants) == manifest["variants"], "Workload counts differ")
    group = lambda r: "semantic_silver" if r.meta["judgment"] == "semantic_silver" else "strong"
    ndcg = ("ndcg_at_10",)
    summary = summarize(primary, CI_METRICS)
    for i, side in enumerate(SIDES):
        mapping = {"ndcg_at_10": "ndcg_at_10", "known_recall_at_10": "known_recall_at_10",
                   "mrr_at_10": "mrr_at_10_single", "success_at_1": "success_at_1_single"}
        for local, author in mapping.items():
            confirm(abs(summary["metrics"][local][side] - official[i][author]) <= .0000051,
                    f"{name}: metrics disagree with the publisher's program")
        confirm(summary["scope"]["outside_requested_window"][i] == official[i]["outside_explicit_window_documents"]
                and summary["metrics"]["false_empty_rate"]["events"][i] == official[i]["false_empty"]
                and summary["metrics"]["false_nonempty_rate"]["events"][i] == official[i]["false_nonempty"],
                "Scope or empty-state audit differs from publisher counts")
    by_validation_group = paired_ci(primary, "ndcg_at_10", "validation_group")
    panel_variants = [c for c in variants if c.meta["original_query_id"] in panel]
    return {
        "corpus_sha256": manifest["sha256"]["documents.jsonl"], "corpus_documents": manifest["documents"],
        "suite_manifest_sha256": digest(pool / "suite" / name / "manifest.json"),
        "panel_sha256": digest(pool / "panels" / (name + ".jsonl")), "http_rankings": provenance,
        "primary": summary, "primary_by_judgment_tier": breakdown(primary, group, ndcg),
        "primary_by_task_type": breakdown(primary, lambda r: r.meta["task_type"], ndcg),
        "primary_by_task_family": breakdown(primary, lambda r: r.meta["family"]),
        "primary_by_judgment": breakdown(primary, lambda r: r.meta["judgment"]),
        "primary_paired_by_validation_group": {"metric": "ndcg_at_10", **by_validation_group},
        "variants": {"all": summarize(variants, ndcg), "by_style": breakdown(variants, lambda r: r.meta["variant"], ndcg),
                     "by_judgment_tier": breakdown(variants, group, ndcg)},
        "panel": {"primary": summarize([cases[qid] for qid in panel], CI_METRICS),
                  "primary_by_judgment_tier": breakdown([cases[qid] for qid in panel], group, ndcg),
                  "variants": summarize(panel_variants, ndcg),
                  "variant_by_style": breakdown(panel_variants, lambda r: r.meta["variant"], ndcg)},
    }


def verify_endpoints(pool, runs, urls):
    """Check original bodies from released positive queries only, including legacy HTTP servers."""
    from urllib.parse import quote, urlencode
    from urllib.request import urlopen

    for name in urls:
        ds = pool / "suite" / name
        documents = {row["doc_id"]: row for row in lines(ds / "documents.jsonl")}
        queries = keyed(lines(ds / "queries_dev.jsonl"), "positive queries")
        by_box = defaultdict(dict)
        for row in lines(ds / "qrels_dev.jsonl"):
            if row["relevance"] > 0:
                q = queries[row["query_id"]]
                by_box[q["synthetic_user"]].setdefault(row["doc_id"], q["query_date"])
        for side in SIDES:
            count = 0
            for box, group in by_box.items():
                ids = sorted(group)
                for did in sorted({ids[0], ids[len(ids)//2], ids[-1]}):
                    path = "/api/mail/" + quote(did, safe="") + "?" + urlencode({"mailbox": box, "date": group[did]})
                    with urlopen(urls[name][side].rstrip("/") + path, timeout=40) as response:
                        item = json.load(response)
                    confirm(item["id"] == did and item["body"] == documents[did]["doc"],
                            "Original HTTP body differs from the suite")
                    count += 1
            stamp = {"corpus_sha256": digest(ds / "documents.jsonl"),
                     "open_benchmark_mailboxes_sampled": len(by_box), "original_bodies_verified": count}
            (runs / name / side / "http_samples.json").write_text(json.dumps(stamp) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=ROOT / ".cache/ood-mail")
    parser.add_argument("--output", type=Path, default=ROOT / "results/ood_mail_retrieval.json")
    parser.add_argument("--baseline-revision", default=BEFORE)
    parser.add_argument("--candidate-revision", default=AFTER)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev",
                        help="Both splits share one byte-identical corpus and server per mail source.")
    parser.add_argument("--verify-servers", action="store_true", help="Read a sample of released target documents via HTTP.")
    parser.add_argument("--workers", type=int, default=2)
    ports = {"vk_dev": [18773, 8766], "personal_dev": [18777, 18776]}
    for name in SUITES:
        for i, side in enumerate(SIDES):
            parser.add_argument(f"--{name.replace('_','-')}-{side}", default=f"http://127.0.0.1:{ports[name][i]}")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    os.umask(0o077)
    revisions = {"before": commit_id(args.baseline_revision), "after": commit_id(args.candidate_revision)}
    suites = [f"{source}_{split}" for split in (("dev", "holdout") if args.split == "all" else (args.split,))
              for source in ("vk", "personal")]
    urls = {name: {side: getattr(args, f"{name.split('_')[0]}_dev_{side}") for side in SIDES} for name in suites}
    if args.run or args.verify_servers:
        for endpoint in (args.personal_dev_before, args.personal_dev_after):
            confirm(urlparse(endpoint).hostname in {"localhost", "127.0.0.1", "::1"},
                    "Personal-mail queries and source checks require a loopback endpoint.")
    manifests = {name: check_inputs(args.pool, name) for name in suites}
    if args.run:
        args.runs.mkdir(parents=True, exist_ok=True)
        run_http(args.pool, args.runs, urls, args.workers)
    if args.run or args.verify_servers:
        verify_endpoints(args.pool, args.runs, urls)
    datasets = {name: results(args.pool, args.runs, name, manifests[name], revisions) for name in suites}
    for name in suites:
        for side in SIDES:
            proof = archive_evidence(args.runs, urls[name][side], revisions[side],
                                     datasets[name]["corpus_sha256"])
            if proof:
                datasets[name]["http_rankings"][side]["verified_local_launch"] = proof
            path = args.runs / name / side / "http_samples.json"
            if path.exists():
                stamp = json.loads(path.read_bytes())
                confirm(stamp["corpus_sha256"] == datasets[name]["corpus_sha256"], "Sample snapshot mismatch")
                datasets[name]["http_rankings"][side]["sampled_original_bodies"] = stamp["original_bodies_verified"]
                datasets[name]["http_rankings"][side]["sampled_open_mailboxes"] = stamp["open_benchmark_mailboxes_sampled"]
    report = {
        "protocol": {"suite_revision": SOURCE, "baseline_revision": revisions["before"], "candidate_revision": revisions["after"],
                     "http_endpoint": "POST /api/search", "repeats": 1, "depth": 10,
                     "scope_order": ["before", "after"], "target": args.split, "suites": suites,
                     "ranking": "Primary and variant queries were requested on both revisions. Panel metrics use the declared subset of full-run rankings. A separate HTTP panel repeat, when present, must return every ranking identically.",
                     "judgments": "Strong combines exact rules and assistant-reviewed windows. Semantic silver uses incomplete category rules. Unjudged results give no ranking gain.",
                     "metrics": "Query macro means. nDCG uses exponential gain. Known recall counts all graded positives. Singleton MRR uses exactly one positive judgment. Author-target MRR and success use designated known-item targets. Judged@10 is the judged fraction of returned documents on positive, nonempty rankings.",
                     "scope": "Mailbox, unknown ID and as-of violations are counted across all returned top-10 IDs. The separate requested receipt window counts undated and out-of-window hits, on positives and verified empties.",
                     "uncertainty": "Paired percentile cluster bootstrap, 10,000 draws, seed 2026. Resample family_id with all eligible queries; no variant enters primary. A separate positive-query interval groups by publisher validation_group. One owner yields no between-owner interval; few mailbox groups give descriptive intervals only.",
                     "snapshot": "Candidate reports server SHA-256; legacy HTTP exposes mailbox counts. Per-server provenance records additional local source/corpus verification and sampled original HTTP bodies when available."},
        "datasets": datasets}
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for name, ds in datasets.items():
        print(name, ds["primary"]["queries"], ds["primary"]["metrics"]["ndcg_at_10"])


if __name__ == "__main__":
    main()
