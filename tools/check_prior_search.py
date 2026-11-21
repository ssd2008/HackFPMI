"""Repeat historical VK search checks and, optionally, two local export probe suites.

The two server stamps are JSON files with revision, port, source_archive_sha256 and
corpus_sha256 for running VK APIs. Private export cases run in an isolated process
against the same commit's search and direct-read endpoints. Raw ranks stay in --cache.
"""
import argparse
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import date
import hashlib
import io
import json
import os
from pathlib import Path
from statistics import fmean
import subprocess
import sys
import tarfile
import traceback
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlsplit

import check_search_probes as probes
import validate_rankings as validation

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests/fixtures"
PILOT = FIX / "validation_pilot"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def archive(revision):
    return subprocess.check_output(["git", "-C", str(ROOT), "archive", revision, "mailsearch", "static"])


def check_stamp(path):
    from check_mail_ood import archive_evidence

    stamp = read(path)
    rev = stamp["revision"]
    resolved = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--verify", "--end-of-options", rev + "^{commit}"], text=True).strip()
    if resolved != rev or hashlib.sha256(archive(rev)).hexdigest() != stamp["source_archive_sha256"]:
        raise ValueError("Server stamp does not match the committed application archive")
    base = "http://127.0.0.1:" + str(int(stamp["port"]))
    source = archive_evidence(path.parent, base, rev, stamp["corpus_sha256"])
    if source is None:
        raise ValueError("The stamp and extracted server source tree must share a run directory")
    status, live = probes.request_json(base, "/api/mailboxes")
    if status != 200 or live["corpus_sha256"] != stamp["corpus_sha256"] or live["corpus_sha256"] != sha(ROOT / "data/documents.jsonl"):
        raise ValueError("Expected the pinned VK corpus")
    return rev, base, live


def probe_run(fixture, documents, base, request=None):
    suite = read(fixture)
    docs = probes.load_corpus(documents, suite["documents_sha256"])
    probes.verify_labels(suite, docs)
    original = probes.request_json
    if request:
        probes.request_json = request
    try:
        rows = [probes.check_search(base, c, docs) for c in suite["cases"]]
        reads = [probes.check_read(base, c, docs) for c in suite.get("read_checks", [])]
    finally:
        probes.request_json = original
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    return {"fixture_sha256": sha(fixture), "documents_sha256": suite["documents_sha256"],
            "metrics": probes.summarize(rows, reads),
            "by_family": {k: probes.summarize(v, []) for k, v in sorted(by_family.items())}}


def private_worker(rev, dest, folders, smoke):
    package = archive(rev)
    engine_dir = dest / "engine"
    engine_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(package)) as src:
        src.extractall(engine_dir, filter="data")
    if "mailsearch" in sys.modules:
        raise RuntimeError("An application module was imported before selecting the archived revision")
    sys.path.insert(0, str(engine_dir))
    from fastapi import HTTPException
    from mailsearch import app as api
    from mailsearch.retrieval import SearchEngine
    from mailsearch.store import MailStore

    pinned = (engine_dir / "mailsearch").resolve()
    for module in (api, sys.modules[SearchEngine.__module__], sys.modules[MailStore.__module__]):
        if Path(module.__file__).resolve().parent != pinned:
            raise RuntimeError("An endpoint or index module came from the wrong Git archive")

    def local(_base, endpoint, body=None):
        if endpoint == "/api/search":
            response = api.search(api.SearchRequest(**body))
            return (response.status_code, json.loads(response.body)) if hasattr(response, "body") else (200, response)
        parsed = urlsplit(endpoint)
        params = parse_qs(parsed.query)
        try:
            return 200, api.message(unquote(parsed.path.removeprefix("/api/mail/")),
                                    params["mailbox"][0], date.fromisoformat(params["date"][0]))
        except HTTPException as error:
            return error.status_code, None

    out = {}
    for num, folder in enumerate(folders, 1):
        documents = folder / "corpus/documents.jsonl"
        docs = lines(documents)
        api.app.state.store = MailStore(docs)
        api.app.state.search_engine = SearchEngine(docs, ROOT / ".cache/mailsearch")
        for label, fn in (("original", "probes.json"), ("quoted_followup", "fresh-quotes.json")):
            out[f"export_{num}_{label}"] = probe_run(folder / fn, documents, "", local)
    save(dest / "private.json", out)
    if smoke:
        documents = lines(smoke / "documents.jsonl")
        catalog = {d["doc_id"]: d for d in documents}
        queries = lines(smoke / "queries_dev.jsonl")
        labels = defaultdict(dict)
        by_id = {q["query_id"]: q for q in queries}
        if len(queries) != len(by_id) or len(documents) != len(catalog):
            raise ValueError("The wiring fixture contains duplicate IDs")
        for row in lines(smoke / "qrels_dev.jsonl"):
            qid, did, grade = row["query_id"], row["doc_id"], row["relevance"]
            q = by_id[qid]
            context = {"mailbox": q["synthetic_user"], "date": q["query_date"]}
            if did in labels[qid] or type(grade) is not int or grade < 0 or not probes.visible(catalog[did], context):
                raise ValueError("Invalid wiring-fixture qrel")
            labels[qid][did] = grade
        api.app.state.store = MailStore(documents)
        api.app.state.search_engine = SearchEngine(documents, ROOT / ".cache/mailsearch")
        hits, judged, returned = 0, 0, 0
        for q in queries:
            r = {"mailbox": q["synthetic_user"], "date": q["query_date"], "query": q["query"], "limit": 10}
            selected = [item["id"] for item in local("", "/api/search", r)[1]["items"]]
            assert len(selected) == len(set(selected)) <= 10
            assert all(probes.visible(catalog[did], r) for did in selected)
            known = {d for d, g in labels[q["query_id"]].items() if g > 0}
            judged += bool(known)
            hits += bool(known.intersection(selected))
            returned += len(selected)
        save(dest / "smoke.json", {"input_sha256": {n: sha(smoke / n) for n in ("documents.jsonl", "queries_dev.jsonl", "qrels_dev.jsonl")},
                                   "documents": len(documents), "queries": len(queries), "judged_positive": judged,
                                   "unjudged": len(queries) - judged, "positive_hit_at_10": hits,
                                   "returned": returned, "scope_violations": 0})


def run_vk(stamp, cache, spelling):
    sys.path.insert(0, str(ROOT))
    import benchmark
    rev, base, live = check_stamp(stamp)
    dest = cache / rev
    dest.mkdir(parents=True, exist_ok=True)
    sets = {"paraphrased": (FIX / "queries_paraphrased.jsonl", None, False),
            "typos": (FIX / "queries_typos.jsonl", None, False),
            "no_date": (FIX / "queries_no_date.jsonl", None, True),
            "pilot": (PILOT / "queries.jsonl", PILOT / "qrels.jsonl", False)}
    if spelling:
        rows = [dict(query_id=str(i), query=q, synthetic_user=live["default_mailbox"], query_date=live["default_date"])
                for i, q in enumerate(read(spelling))]
        query_file = dest / "spelling_stress_queries.jsonl"
        query_file.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in rows))
        sets["spelling_stress"] = query_file, None, True
    for key, (queries, qrels, unjudged) in sets.items():
        with (dest / (key + ".log")).open("w") as log, redirect_stdout(log):
            benchmark.run(SimpleNamespace(data=ROOT / "data", queries=queries, qrels=qrels, unjudged=unjudged,
                                          url=base, k=[1, 5, 10], repeats=1, concurrency=1, output=dest / (key + ".json")))
    result = {n: probe_run(FIX / (n + ".json"), ROOT / "data/documents.jsonl", base)
              for n in ("search_probes", "search_probes_extra")}
    save(dest / "public_probes.json", result)
    return rev, dest


def describe_report(report):
    lengths = [len(row["ranked_ids"]) for row in report["queries"].values()]
    return {"queries": len(lengths), "nonempty": sum(bool(n) for n in lengths),
            "returned": sum(lengths), "scope_violations": 0,
            "quality_at_10": report["quality_first_pass"].get("10")}


def paired_reports(a, b):
    left, right = read(a), read(b)
    assert left["dataset"]["sha256"] == right["dataset"]["sha256"]
    assert left["queries"].keys() == right["queries"].keys()
    assert left["dataset"]["server_corpus_check"] == right["dataset"]["server_corpus_check"] == "sha256"
    compared = [(v["ranked_ids"], right["queries"][k]["ranked_ids"]) for k, v in left["queries"].items()]
    return {"input_sha256": left["dataset"]["sha256"], "baseline": describe_report(left),
            "current": describe_report(right), "changed_lists": sum(x != y for x, y in compared),
            "mean_shared_ids": round(fmean(len(set(x) & set(y)) for x, y in compared), 6)}


def pair_probes(left, right):
    if set(left) != set(right):
        raise ValueError("Probe suites differ")
    out = {}
    for name, a in left.items():
        b = right[name]
        assert a["fixture_sha256"] == b["fixture_sha256"] and a["documents_sha256"] == b["documents_sha256"]
        assert a["by_family"].keys() == b["by_family"].keys()
        grouped = {}
        for family, x in a["by_family"].items():
            y = b["by_family"][family]
            if x["positive_queries"] + x["empty_queries"] < 2:
                continue
            grouped[family] = {"positive": x["positive_queries"], "empty": x["empty_queries"],
                               **{k: [x[k], y[k]] for k in ("hit_at_1", "hit_at_5", "correctly_empty")}}
        out[name] = {"fixture_sha256": a["fixture_sha256"], "documents_sha256": a["documents_sha256"],
                     "baseline": a["metrics"], "current": b["metrics"], "families_with_at_least_two_queries": grouped}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before-stamp", type=Path)
    parser.add_argument("--after-stamp", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--private-exports", type=Path, nargs="*", default=[])
    parser.add_argument("--spelling-controls", type=Path)
    parser.add_argument("--smoke-corpus", type=Path, help="Optional historical three-file wiring fixture")
    parser.add_argument("--private-worker", metavar="REVISION", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.umask(0o077)
    if args.private_worker:
        try:
            private_worker(args.private_worker, args.cache, args.private_exports, args.smoke_corpus)
        except Exception:
            args.cache.mkdir(parents=True, exist_ok=True)
            (args.cache / "private-validation-error.log").write_text(traceback.format_exc())
            sys.exit("Private validation failed. Details are in the private run directory.")
        return
    if not args.before_stamp or not args.after_stamp or not args.output:
        parser.error("--before-stamp, --after-stamp and --output are required")
    versions = [run_vk(stamp, args.cache, args.spelling_controls) for stamp in (args.before_stamp, args.after_stamp)]
    (before, a), (after, b) = versions
    report = {"revisions": {"baseline": before, "current": after},
              "server_archive_sha256": {"baseline": read(args.before_stamp)["source_archive_sha256"],
                                        "current": read(args.after_stamp)["source_archive_sha256"]},
              "protocol": {"depth": 10, "public_execution": "Pinned localhost HTTP, corpus SHA-256 and mailbox/as-of checks.",
                           "private_execution": "Pinned production search and direct-read endpoint functions on each separate frozen export.",
                           "variants": "Known rewrites of the same 99 dated development queries. qrels used for paraphrases and typos only.",
                           "unjudged": "Removing the requested date changes intent. Results without dates and spelling stress controls have no relevance or empty labels.",
                           "probes": "Hit@1/5 on confirmed anchors, empty counts on stated absence rules. Other hits are unjudged. Scope and direct reads audited separately.",
                           "pilot": "Existing agent-graded pilot, six development and six historical blind tasks. This round reuses the original splits; it is not a new human blind test."},
              "public_variants": {k: paired_reports(a / (k + ".json"), b / (k + ".json"))
                                  for k in ("paraphrased", "typos", "no_date", *(["spelling_stress"] if args.spelling_controls else []))},
              "vk_probes": pair_probes(read(a / "public_probes.json"), read(b / "public_probes.json"))}
    if args.spelling_controls:
        report["public_variants"]["spelling_stress"]["source_list_sha256"] = sha(args.spelling_controls)
    pilot = validation.compare(a / "pilot.json", b / "pilot.json", PILOT / "queries.jsonl", PILOT / "qrels.jsonl",
                               PILOT / "tasks.jsonl")
    report["pilot"] = {"inputs": pilot["inputs"], "protocol": pilot["protocol"], "groups": pilot["primary"]}
    if args.private_exports or args.smoke_corpus:
        for revision, dest in versions:
            subprocess.run([sys.executable, __file__, "--private-worker", revision, "--cache", str(dest), "--private-exports",
                            *map(str, args.private_exports), *(["--smoke-corpus", str(args.smoke_corpus)] if args.smoke_corpus else [])],
                           check=True, env={**os.environ, "HF_HUB_OFFLINE": "1",
                                                                              "TRANSFORMERS_OFFLINE": "1", "OPENBLAS_NUM_THREADS": "1"})
        if args.private_exports:
            report["private_probes"] = pair_probes(read(a / "private.json"), read(b / "private.json"))
        if args.smoke_corpus:
            report["wiring_smoke"] = {"baseline": read(a / "smoke.json"), "current": read(b / "smoke.json")}
            assert report["wiring_smoke"]["baseline"]["input_sha256"] == report["wiring_smoke"]["current"]["input_sha256"]
    save(args.output, report)
    args.output.chmod(0o644)


if __name__ == "__main__":
    main()
