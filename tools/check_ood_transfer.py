"""Evaluate supplied external retrieval packages, in place and entirely offline.

    .venv/bin/python tools/check_ood_transfer.py --benchmarks /path/to/external/benchmarks

The default covers Enron, SciFact and RuBQ. --datasets selects any of the nine packages.
Use --prepare-only to validate and freeze all inputs before ranking.
Optional --prior-rubq-cache points at the author-Git run from tools/check_rubq.py.
Use another --cache/--output with --baseline-revision and --current-revision to compare other local commits.
--reuse-cache points at a previous comparison. Legacy runs also need its published --reuse-report.
Prior rankings retain their original checksums and remain in their source directory.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date
from email.utils import parseaddr
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
from statistics import fmean
import subprocess
import sys
import tarfile
import unicodedata

from check_rubq import read_lines, write_json


ROOT = Path(__file__).resolve().parents[1]
BASE = "5fe3f5d3739ac523d0b4088f15eef4b50391dd28"
CURRENT = "b37c309ff3d9e83f80a726325a3b9b0ee56c33c1"
NAMES = ("beir-scifact", "enron-mail", "rubq-2.0")
ALL_NAMES = (*NAMES, "miracl-ru", "mr-tydi-ru", "msmarco-doc", "msmarco-passage", "mmarco-ru", "lotte")
CHANNELS = {"enron-mail": ("dev", "empty", "variants")}
MAIN_FILES = ("documents.jsonl", "queries_dev.jsonl", "qrels_dev.jsonl")
SELECTION = "Every supplied document in its original order and every supplied query in the listed channels. Both exact Git revisions use production SearchEngine.search with k=10, original mailbox, query date and text. No RAG generation or tuning."


def checksum(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def channels(name):
    return CHANNELS.get(name, ("dev",))


def resolve_revision(value):
    result = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--verify", "--end-of-options",
                                      f"{value}^{{commit}}"], text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", result):
        raise ValueError("Expected a single local Git commit.")
    return result


def read_inputs(folder):
    docs = read_lines(folder / "documents.jsonl")
    scope = {d["doc_id"]: (d["synthetic_user"], d["doc_dt"]) for d in docs}
    if len(scope) != len(docs):
        raise ValueError("Duplicate corpus ID.")
    queries, labels = {}, {}
    for ch in channels(folder.name):
        queries[ch] = read_lines(folder / f"queries_{ch}.jsonl")
        by_id = {q["query_id"]: q for q in queries[ch]}
        if len(by_id) != len(queries[ch]):
            raise ValueError("Duplicate query ID.")
        labels[ch] = {qid: {} for qid in by_id}
        for row in read_lines(folder / f"qrels_{ch}.jsonl"):
            qid, did, grade = row["query_id"], row["doc_id"], row["relevance"]
            box, day = scope[did]
            q = by_id[qid]
            if (box != q["synthetic_user"] or (day and day > q["query_date"])
                    or type(grade) is not int or grade < 0 or did in labels[ch][qid]):
                raise ValueError("Invalid or duplicate supplied judgment, mailbox or as-of date.")
            labels[ch][qid][did] = grade
        for q in queries[ch]:
            date.fromisoformat(q["query_date"])
            has_positive = any(g > 0 for g in labels[ch][q["query_id"]].values())
            if (ch == "empty") == has_positive:
                raise ValueError("A supplied query has inconsistent channel and positive labels.")
    return docs, scope, queries, labels


def freeze(bench, names=NAMES, engines=("baseline", "current"), *, baseline_revision=BASE, current_revision=CURRENT):
    inputs = {}
    for name in names:
        folder = bench / name
        manifest = json.loads((folder / "manifest.json").read_bytes())
        hashes = {f: checksum(folder / f) for f in manifest["sha256"]}
        if hashes != manifest["sha256"] or not set(MAIN_FILES) <= hashes.keys():
            raise ValueError("Supplied manifest checksum mismatch: " + name)
        docs, scope, queries, labels = read_inputs(folder)
        counts = {key: len(v) for key, v in queries.items()}
        qrels = {ch: sum(map(len, per_query.values())) for ch, per_query in labels.items()}
        if (len(docs), counts["dev"], qrels["dev"]) != (manifest["documents"], manifest["queries"], manifest["qrels"]):
            raise ValueError("Unexpected manifest counts for " + name)
        snapshot = {"manifest_sha256": checksum(folder / "manifest.json"), "sha256": hashes, "documents": len(docs),
                    "mailboxes": len({box for box, day in scope.values()}), "dated_documents": sum(bool(day) for box, day in scope.values()),
                    "queries": counts, "qrels": qrels}
        if name == "enron-mail":
            archive = json.loads((folder / "archive.json").read_bytes())
            if archive["documents_sha256"] != hashes["documents.jsonl"]:
                raise ValueError("Enron archive metadata and manifest describe different snapshots.")
            snapshot["archive"] = {k: archive[k] for k in ("source_sha256", "archive_message_files", "selected_message_files",
                                                            "within_mailbox_duplicates", "unknown_dates")}
            snapshot["owner_written_or_manually_reviewed_primaries"] = manifest["reviewed_primary"]
        else:
            snapshot.update(full_source_corpus=manifest["complete_corpus"], language=manifest["language"])
            positives = {did for ls in labels["dev"].values() for did, grade in ls.items() if grade > 0}
            snapshot["all_ever_positive_ids_in_corpus_prefix"] = set(scope.keys()) >= positives and {
                row["doc_id"] for row in docs[:len(positives)]} == positives
        inputs[name] = snapshot
    commits = {"baseline": resolve_revision(baseline_revision), "current": resolve_revision(current_revision)}
    if len(engines) == 2 and commits["baseline"] == commits["current"]:
        raise ValueError("Paired comparison requires two distinct revisions.")
    selection = SELECTION if len(engines) == 2 else SELECTION.replace("Both exact Git revisions use", "The selected exact Git revision uses")
    return {"engines": {k: commits[k] for k in engines}, "selection": selection, "cutoff": 10, "inputs": inputs}


def verify_same_task(source, target, name, revision):
    single = SELECTION.replace("Both exact Git revisions use", "The selected exact Git revision uses")
    if (revision not in source["engines"].values() or revision not in target["engines"].values()
            or source["inputs"].get(name) != target["inputs"][name]
            or source["selection"] not in (SELECTION, single) or target["selection"] not in (SELECTION, single)
            or source["cutoff"] != target["cutoff"]):
        raise ValueError("The prior engine, document/query/labels snapshot or search protocol differs.")


def same_dense_source(first, second):
    return not subprocess.check_output(["git", "-C", str(ROOT), "diff", "--name-only", first, second,
                                        "--", "mailsearch/dense.py", "mailsearch/__init__.py"])


def vector_directory(cache, name, revision, protocol):
    commits = list(protocol["engines"].values())
    directory = cache / "vectors" / name
    # The original frozen comparison explicitly shared passage vectors while changing the query encoder.
    if set(commits) == {BASE, CURRENT} or all(same_dense_source(commits[0], other) for other in commits[1:]):
        return directory
    return directory / revision


def ranking_artifact(cache, name, revision, target, freeze_sha):
    """Resolve a separately recorded origin without rewriting its original freeze or ranks."""
    local = cache / name / f"{revision}.json"
    pointer = local.with_suffix(".origin.json")
    provenance = None
    expected_freeze = freeze_sha
    if pointer.is_file():
        origin = json.loads(pointer.read_bytes())
        if origin["target_freeze_sha256"] != freeze_sha or origin["engine_revision"] != revision or origin["dataset"] != name:
            raise ValueError("Prior-ranking reference belongs to a different experiment.")
        source_cache = (cache / origin["relative_cache"]).resolve()
        old_raw = (source_cache / "freeze.json").read_bytes()
        expected_freeze = hashlib.sha256(old_raw).hexdigest()
        if expected_freeze != origin["source_freeze_sha256"]:
            raise ValueError("The source comparison freeze changed.")
        source = json.loads(old_raw)
        verify_same_task(source, target, name, revision)
        if source["engines"].get(origin["source_engine_role"]) != revision:
            raise ValueError("The prior engine role does not identify the requested commit.")
        local = source_cache / name / f"{revision}.json"
        provenance = {k: origin[k] for k in ("source_freeze_sha256", "source_artifact_sha256", "source_engine_role",
                                             "source_report_sha256", "revision_evidence")}
    raw = local.read_bytes()
    if provenance and hashlib.sha256(raw).hexdigest() != provenance["source_artifact_sha256"]:
        raise ValueError("The reused ranking artifact changed.")
    run = json.loads(raw)
    if (run["freeze_sha256"] != expected_freeze or run.get("dataset", name) != name
            or run.get("engine_revision", revision) != revision):
        raise ValueError("Ranking file does not match its actual originating freeze or engine.")
    return run, provenance


def verified_legacy_report(reports, source, old_sha, name, revision, run, queries, labels):
    """Old caches only encoded the commit in the filename. Check its published per-engine scores too."""
    role = next(role for role, rev in source["engines"].items() if rev == revision)
    for path in reports:
        published = json.loads(path.read_bytes())
        if published.get("frozen_sha256") != old_sha or published.get("protocol") != source:
            continue
        section = published["datasets"][name]
        vectors_sha = section.get("passage_vectors_sha256") or section["passage_vectors_sha256_by_engine"][role]
        if vectors_sha != run["vectors_sha256"]:
            raise ValueError("Legacy passage vectors differ from the published report.")
        for ch, qs in queries.items():
            values = {q["query_id"]: scores(run["ranked_ids"][ch][q["query_id"]], labels[ch][q["query_id"]]) for q in qs}
            metrics = summary(list(values), {role: values})[role]
            published_metrics = section["channels"][ch][role]
            if any(published_metrics.get(key) != value for key, value in metrics.items()):
                raise ValueError("Legacy ranking disagrees with the published aggregate for this revision.")
        return checksum(path)
    raise ValueError("Legacy ranking lacks embedded revision metadata. Supply its original aggregate with --reuse-report.")


def register_reuse(bench, cache, source_cache, reports, target):
    from filelock import FileLock

    if source_cache == cache:
        raise ValueError("The destination and source caches must be different.")
    source_raw = (source_cache / "freeze.json").read_bytes()
    source, old_sha = json.loads(source_raw), hashlib.sha256(source_raw).hexdigest()
    target_sha = checksum(cache / "freeze.json")
    common = set(source["engines"].values()) & set(target["engines"].values())
    for name in target["inputs"].keys() & source["inputs"].keys():
        matches = [rev for rev in sorted(common) if (source_cache / name / f"{rev}.json").is_file()]
        if not matches:
            continue
        _, scope, queries, labels = read_inputs(bench / name)
        for revision in matches:
            verify_same_task(source, target, name, revision)
            original = source_cache / name / f"{revision}.json"
            raw = original.read_bytes()
            run = json.loads(raw)
            if (run["freeze_sha256"] != old_sha or run.get("engine_revision", revision) != revision
                    or run.get("dataset", name) != name or set(run["ranked_ids"]) != set(queries)):
                raise ValueError("Source rank does not match the requested version or source freeze.")
            for ch, qs in queries.items():
                ranks = run["ranked_ids"][ch]
                if set(ranks) != {q["query_id"] for q in qs}:
                    raise ValueError("Source rankings omit frozen questions.")
                for q in qs:
                    ids = ranks[q["query_id"]]
                    if not isinstance(ids, list) or len(ids) > target["cutoff"] or len(set(ids)) != len(ids):
                        raise ValueError("Invalid prior ranked IDs.")
                    for did in ids:
                        box, day = scope[did]
                        if box != q["synthetic_user"] or (day and day > q["query_date"]):
                            raise ValueError("Source ranking failed the mailbox/as-of check.")
            report_sha = None
            evidence = "revision embedded in source run and named by source freeze"
            if "engine_revision" not in run:
                report_sha = verified_legacy_report(reports, source, old_sha, name, revision, run, queries, labels)
                evidence = "full-commit filename, source freeze, passage vectors and all primary metrics matched the published engine role"
            binding = {"dataset": name, "engine_revision": revision, "relative_cache": os.path.relpath(source_cache, cache),
                       "target_freeze_sha256": target_sha, "source_freeze_sha256": old_sha,
                       "source_artifact_sha256": hashlib.sha256(raw).hexdigest(), "source_report_sha256": report_sha,
                       "source_engine_role": next(k for k, v in source["engines"].items() if v == revision),
                       "revision_evidence": evidence}
            dest = cache / name / f"{revision}.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(dest) + ".lock"):
                if dest.exists():
                    continue  # Preserve results independently computed in this destination.
                pointer = dest.with_suffix(".origin.json")
                if pointer.is_file() and json.loads(pointer.read_bytes()) != binding:
                    raise ValueError("A different prior source is already registered for this result.")
                write_json(pointer, binding)
            # Match the actual bytes and encoder before offering an immutable passage index.
            old_vectors = vector_directory(source_cache, name, revision, source)
            for source_index in old_vectors.glob("*.npy"):
                if checksum(source_index) != run["vectors_sha256"]:
                    continue
                for rev in target["engines"].values():
                    if not same_dense_source(revision, rev):
                        continue
                    base = vector_directory(cache, name, rev, target)
                    base.mkdir(parents=True, exist_ok=True)
                    linked = base / source_index.name
                    if not linked.exists():
                        try:
                            os.link(source_index, linked)
                        except OSError:
                            pass  # Across filesystems the engine builds its normal content-keyed local index.
                    if linked.exists() and checksum(linked) != run["vectors_sha256"]:
                        raise ValueError("Destination passage index conflicts with the verified source.")
                break


def rank(bench, cache, name, revision):
    from filelock import FileLock

    path = bench / name
    frozen = json.loads((cache / "freeze.json").read_bytes())
    freeze_sha = checksum(cache / "freeze.json")
    if revision not in frozen["engines"].values():
        raise ValueError("Revision is outside this cache's frozen comparison.")
    for filename, expected in frozen["inputs"][name]["sha256"].items():
        if filename.startswith(("documents", "queries")) and checksum(path / filename) != expected:
            raise ValueError("Inputs changed since freeze.")
    result_path = cache / name / f"{revision}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(result_path) + ".lock"):
        if result_path.with_suffix(".origin.json").is_file():
            ranking_artifact(cache, name, revision, frozen, freeze_sha)
            print(f"Using verified prior {name} {revision[:7]}", flush=True)
            return
        if result_path.is_file():
            try:
                saved = json.loads(result_path.read_bytes())
                complete = (saved["freeze_sha256"] == freeze_sha and len(bytes.fromhex(saved["vectors_sha256"])) == 32
                            and saved.get("engine_revision", revision) == revision and saved.get("dataset", name) == name)
                complete = complete and set(saved["ranked_ids"]) == set(channels(name))
                for ch in channels(name):
                    expected = {q["query_id"] for q in read_lines(path / f"queries_{ch}.jsonl")}
                    ranks = saved["ranked_ids"][ch]
                    complete = complete and isinstance(ranks, dict) and set(ranks) == expected and all(
                        isinstance(ids, list) and len(ids) <= 10 and all(isinstance(did, str) for did in ids)
                        and len(set(ids)) == len(ids) for ids in ranks.values())
                if complete:
                    print(f"Using completed {name} {revision[:7]}", flush=True)
                    return
            except (ValueError, TypeError, KeyError):
                pass  # Rebuild malformed cached output from the unchanged inputs.
        _compute_ranking(path, cache, name, revision, freeze_sha, result_path)


def _compute_ranking(path, cache, name, revision, freeze_sha, result_path):
    dest = cache / "engines" / name / revision
    dest.mkdir(parents=True, exist_ok=True)
    package = subprocess.check_output(["git", "-C", str(ROOT), "archive", revision, "mailsearch"])
    with tarfile.open(fileobj=io.BytesIO(package)) as archive:
        archive.extractall(dest, filter="data")
    sys.path.insert(0, str(dest))
    from mailsearch.retrieval import SearchEngine

    # The retrieval process never opens a file of relevance labels or case descriptions.
    docs = read_lines(path / "documents.jsonl")
    protocol = json.loads((cache / "freeze.json").read_bytes())
    engine = SearchEngine(docs, vector_directory(cache, name, revision, protocol))
    envelope = {d["doc_id"]: (d["synthetic_user"], d["doc_dt"]) for d in docs}
    results = {}
    for ch in channels(name):
        results[ch] = {}
        for q in read_lines(path / f"queries_{ch}.jsonl"):
            ranks = engine.search(q["synthetic_user"], q["query"], date.fromisoformat(q["query_date"]), limit=10)
            ids = [did for did, score in ranks]
            if len(ids) > 10 or len(set(ids)) != len(ids):
                raise ValueError("Invalid ranking depth or duplicated returned ID.")
            for did in ids:
                box, day = envelope[did]
                if box != q["synthetic_user"] or day and day > q["query_date"]:
                    raise ValueError("Returned document violates mailbox or as-of date.")
            results[ch][q["query_id"]] = ids
        print(f"{name} {revision[:7]} {ch}: {len(results[ch])} queries", flush=True)
    write_json(result_path,
               {"dataset": name, "engine_revision": revision, "freeze_sha256": freeze_sha,
                "vectors_sha256": checksum(Path(engine.dense.vectors.filename)),
                "ranked_ids": results})


def scores(ranks, labels):
    relevant = {did for did, grade in labels.items() if grade > 0}
    result = {"returned": len(ranks)}
    if not relevant:
        return result
    hits = [i for i, did in enumerate(ranks, 1) if did in relevant]
    gain = sum((2 ** labels.get(did, 0) - 1) / math.log2(i + 1) for i, did in enumerate(ranks, 1))
    ideal = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(sorted(labels.values(), reverse=True)[:10], 1))
    result.update(ndcg_at_10=gain / ideal, known_recall_at_10=len(hits) / len(relevant),
                  hit_at_10=int(bool(hits)), mrr_at_10=1 / hits[0] if hits else 0.0)
    return result


def summary(query_ids, evaluations):
    result = {"queries": len(query_ids)}
    for engine, rows in evaluations.items():
        chosen = [rows[qid] for qid in query_ids]
        positive = [s for s in chosen if "ndcg_at_10" in s]
        empty = [s for s in chosen if "ndcg_at_10" not in s]
        metrics = {"returned_documents_checked": sum(s["returned"] for s in chosen),
                   "mailbox_or_asof_violations": 0, "empty_lists_on_positive": sum(not s["returned"] for s in positive)}
        if positive:
            for key in ("ndcg_at_10", "known_recall_at_10", "hit_at_10", "mrr_at_10"):
                metrics[key] = round(fmean(s[key] for s in positive), 8)
        if empty:
            count = sum(bool(s["returned"]) for s in empty)
            metrics.update(false_nonempty_count=count, false_nonempty_rate=round(count / len(empty), 8))
        bounded = [s for s in chosen if "outside_requested_window" in s]
        if bounded:
            metrics["requested_period"] = {"queries": len(bounded), "returned_documents": sum(s["returned"] for s in bounded),
                                          "outside_window": sum(s["outside_requested_window"] for s in bounded)}
        result[engine] = metrics
    if "baseline" in evaluations and "current" in evaluations:
        deltas = [evaluations["current"][qid]["ndcg_at_10"] - evaluations["baseline"][qid]["ndcg_at_10"]
                  for qid in query_ids if "ndcg_at_10" in evaluations["baseline"][qid]]
        if deltas:
            result["paired_ndcg_at_10"] = {"delta": round(fmean(deltas), 8), "wins": sum(d > 1e-12 for d in deltas),
                                          "ties": sum(abs(d) <= 1e-12 for d in deltas), "losses": sum(d < -1e-12 for d in deltas)}
    return result


def audit_enron(docs, scope, queries, labels, cases):
    def normalized(subject):
        return " ".join(unicodedata.normalize("NFKC", subject).casefold().replace("ё", "е").split())

    subjects, senders = defaultdict(set), defaultdict(set)
    for doc in docs:
        sender, subject, _ = doc["doc"].split("\n", 2)
        box, did = doc["synthetic_user"], doc["doc_id"]
        subjects[box, normalized(subject.partition(":")[2])].add(did)
        senders[box, parseaddr(sender.partition(":")[2])[1].lower()].add(did)
    checked = Counter()
    for ch, qs in queries.items():
        for q in qs:
            qid, box = q["query_id"], q["synthetic_user"]
            case = cases[qid]
            gold = {did for did, grade in labels[ch][qid].items() if grade > 0}
            if case["state"] != ("verified_empty" if ch == "empty" else "positive") or len(gold) != case["relevant_count"]:
                raise ValueError("Inconsistent Enron case state or labels.")
            if ch == "variants":
                if labels[ch][qid] != labels["dev"][case["original_query_id"]]:
                    raise ValueError("Variant labels differ from its original case.")
                checked["variant_label_sets"] += 1
            elif case["judgment"] == "exact":
                if case["family"] == "sender":
                    address = q["query"].split("email from ", 1)[1].split(" received", 1)[0]
                    candidates = senders[box, address]
                else:
                    subject = q["query"].split('"', 1)[1].rsplit('"', 1)[0]
                    candidates = subjects[box, normalized(subject)]
                window = case.get("receipt_window")
                eligible = {did for did in candidates if not scope[did][1] or scope[did][1] <= q["query_date"]}
                if window:
                    eligible = {did for did in eligible if scope[did][1] and window[0] <= scope[did][1] <= window[1]}
                if eligible != gold:
                    raise ValueError("Enron exact labels failed an exhaustive metadata check.")
                checked["exhaustive_" + ch] += 1
    return dict(checked)


def compare_rubq_adapter(prior, docs, queries, labels, current_revision=CURRENT):
    if not (prior / "freeze.json").is_file():
        return {"comparison": "Prior local cache unavailable. Recreate with tools/check_rubq.py --prepare-only and repeat --report-only."}
    reference = json.loads((prior / "freeze.json").read_bytes())
    for name in ("queries.jsonl", "documents.jsonl", "qrels.jsonl"):
        if checksum(prior / name) != reference["adapter_sha256"][name]:
            raise ValueError("Prior RuBQ files do not match the original freeze.")
    original = {d["doc_id"]: d for d in read_lines(prior / "documents.jsonl")}
    original_queries = {q["query_id"]: q for q in read_lines(prior / "queries.jsonl")}
    prior_qrels = {(r["query_id"], r["doc_id"], r["relevance"]) for r in read_lines(prior / "qrels.jsonl")}
    para_ids = [d["doc_id"].removeprefix("rubq-2.0/") for d in docs]
    qids = [q["query_id"].removeprefix("rubq-2.0/q/") for q in queries]
    qrels = {(qid.removeprefix("rubq-2.0/q/"), did.removeprefix("rubq-2.0/"), grade)
             for qid, per_query in labels.items() for did, grade in per_query.items()}
    if set(para_ids) != set(original) or set(qids) != set(original_queries):
        return {"comparison": "Different sets of passage or question IDs after stripping the supplied prefix."}
    changes = sum(d["doc"] != original[i]["doc"] for i, d in zip(para_ids, docs))
    comparison = {"author_git_revision": reference["source_revision"], "same_raw_ids_after_removing_namespace": True,
            "same_question_texts": all(q["query"] == original_queries[i]["query"] for i, q in zip(qids, queries)),
            "same_qrels_including_grades": qrels == prior_qrels, "changed_passage_texts": changes,
            "changed_beyond_outer_whitespace": sum(d["doc"].strip() != original[i]["doc"].strip() for i, d in zip(para_ids, docs)),
            "same_document_order": para_ids == list(original), "same_query_order": qids == list(original_queries),
            "author_adapter_query_dates": sorted({q["query_date"] for q in original_queries.values()}),
            "supplied_query_dates": sorted({q["query_date"] for q in queries}),
            "same_receipt_dates": all(d["doc_dt"] == original[i]["doc_dt"] for i, d in zip(para_ids, docs))}
    saved_path = ROOT / "results" / "rubq_transfer.json"
    if saved_path.is_file():
        previous = json.loads(saved_path.read_bytes())
        if previous["dataset"]["adapter_sha256"] != reference["adapter_sha256"]:
            raise ValueError("The saved author-Git report used different inputs.")
        before, after = previous["dataset"]["engine_revisions"]
        modules = ["__init__.py", "dates.py", "dense.py", "retrieval.py", "spelling.py"]
        changes = subprocess.check_output(["git", "-C", str(ROOT), "diff", "--name-only", after, current_revision,
                                          "--", *(f"mailsearch/{m}" for m in modules)])
        comparison["published_author_order_run"] = {"baseline_revision": before, "prior_candidate_revision": after,
            "prior_candidate_retrieval_modules_match_current": not bool(changes),
            "prior_ndcg_at_10": previous["metrics"]["ndcg_at_10"], "report_sha256": checksum(saved_path)}
    return comparison


def report(bench, cache, protocol, prior):
    result = {"protocol": protocol, "metrics": "Positive channels use query-macro graded nDCG@10, recall of known positives, Hit@10 and first-relevant MRR@10. Empty channels use false-nonempty count/rate. Missing labels give zero measured gain.",
              "scope": "All returned IDs were checked against the source mailbox and as-of date. When annotated, requested receipt windows are audited separately.",
              "frozen_sha256": checksum(cache / "freeze.json"), "datasets": {}}
    inventory = bench.parent / "sources.json"
    revisions = json.loads(inventory.read_bytes()).get("revisions", {}) if inventory.is_file() else {}
    repos = {"beir-scifact": ("BeIR/scifact", "BeIR/scifact-qrels"), "rubq-2.0": ("d0rj/RuBQ_2.0", "d0rj/RuBQ_2.0-paragraphs"),
             "miracl-ru": ("miracl/miracl", "miracl/miracl-corpus"), "mr-tydi-ru": ("castorini/mr-tydi", "castorini/mr-tydi-corpus"),
             "mmarco-ru": ("unicamp-dl/mmarco",), "lotte": ("colbertv2/lotte", "colbertv2/lotte_passages")}
    for name in protocol["inputs"]:
        folder = bench / name
        docs, scope, queries, labels = read_inputs(folder)
        parts = {}
        section = {"channels": parts}
        result["datasets"][name] = section
        section["source_revisions"] = {k: revisions[k] for k in repos.get(name, ()) if k in revisions}
        saved = {}
        origins = {}
        for engine, rev in protocol["engines"].items():
            run, provenance = ranking_artifact(cache, name, rev, protocol, result["frozen_sha256"])
            if set(run["ranked_ids"]) != set(queries):
                raise ValueError("Run does not match the frozen package and query channels.")
            saved[engine] = run
            if provenance:
                origins[engine] = {"status": "reused", "revision": rev, **provenance}
        if origins:
            for engine, rev in protocol["engines"].items():
                if engine not in origins:
                    origins[engine] = {"status": "computed", "revision": rev,
                                       "artifact_sha256": checksum(cache / name / f"{rev}.json")}
            section["ranking_provenance"] = origins
        hashes = {r["vectors_sha256"] for r in saved.values()}
        if len(hashes) != 1:
            directories = {vector_directory(cache, name, rev, protocol) for rev in protocol["engines"].values()}
            if len(directories) == 1:
                raise ValueError("The compared engines require identical passage vectors, but their hashes differ.")
            section["passage_vectors_sha256_by_engine"] = {e: run["vectors_sha256"] for e, run in saved.items()}
        else:
            section["passage_vectors_sha256"] = hashes.pop()
        cases = {}
        if name == "enron-mail":
            cases = {r["query_id"]: r for r in read_lines(folder / "cases.jsonl")}
            if set(cases) != {q["query_id"] for qs in queries.values() for q in qs}:
                raise ValueError("Case types do not match supplied query IDs.")
            section["independent_label_audit"] = audit_enron(docs, scope, queries, labels, cases)
        for ch, qs in queries.items():
            ids = [q["query_id"] for q in qs]
            evaluations = {}
            for engine, run in saved.items():
                ranks = run["ranked_ids"][ch]
                if set(ranks) != set(ids):
                    raise ValueError("Run is missing required queries.")
                evals = {}
                for q in qs:
                    qid = q["query_id"]
                    selected = ranks[qid]
                    if len(selected) > 10 or len(selected) != len(set(selected)):
                        raise ValueError("Invalid saved ranking.")
                    for did in selected:
                        box, day = scope[did]
                        if box != q["synthetic_user"] or (day and day > q["query_date"]):
                            raise ValueError("Document scope mismatch in saved results.")
                    evals[qid] = scores(selected, labels[ch][qid])
                    window = cases.get(qid, {}).get("receipt_window")
                    if window:
                        lo, hi = window
                        evals[qid]["outside_requested_window"] = sum(not scope[d][1] or not lo <= scope[d][1] <= hi for d in selected)
                evaluations[engine] = evals
            parts[ch] = summary(ids, evaluations)
            if origins and "baseline" in saved and "current" in saved:
                before, after = (saved[e]["ranked_ids"][ch] for e in ("baseline", "current"))
                different = sum(before[qid] != after[qid] for qid in ids)
                parts[ch]["ranked_lists"] = {"identical": len(ids) - different, "different": different}
            if cases:
                for key in ("judgment", "family"):
                    groups = defaultdict(list)
                    for qid in ids:
                        groups[cases[qid][key]].append(qid)
                    parts[ch]["by_" + key] = {group: summary(qids, evaluations) for group, qids in sorted(groups.items())}
        manifest = json.loads((folder / "manifest.json").read_bytes())
        if name == "enron-mail":
            section["limitations"] = ["Full supplied ten-mailbox snapshot. The CMU archive contains additional mailboxes and duplicate files.",
                                      "Queries and exhaustive sender/subject rules were generated. Semantic labels are subject-based silver judgments. No owner-written or human-reviewed judgments are documented.",
                                      "Empty controls put exact subjects before their first appearance. Variants share their original labels and are summarized separately."]
        else:
            judged = {did for item in labels["dev"].values() for did in item}
            section["raw_corpus_documents"] = manifest["raw_corpus_rows"]
            section["supplied_corpus_composition"] = {k: manifest[k] for k in (
                "documents", "unique_judged_documents", "adjacent_unjudged_documents", "random_unjudged_documents")}
            section["all_judged_docs_in_prefix"] = {d["doc_id"] for d in docs[:len(judged)]} == judged
            section["limitations"] = ["Only supplied judgments are credited; absent labels do not independently establish irrelevance.",
                                      "The adapter places documents judged for any query before random distractors. Stable score ties can depend on that order. Retrieval quality does not measure mail search or generated answers."]
            if not manifest["complete_corpus"]:
                section["limitations"].append("A supplied reduced candidate corpus includes judged documents and unjudged distractors. These scores are conditional on that candidate pool.")
        if name == "rubq-2.0":
            revision = protocol["engines"].get("current", next(iter(protocol["engines"].values())))
            section["author_adapter_comparison"] = compare_rubq_adapter(prior, docs, queries["dev"], labels["dev"], revision)
    return result


def main():
    from filelock import FileLock

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmarks", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=ALL_NAMES, default=list(NAMES),
                        help="Supplied packages to validate and evaluate. Use a separate cache for a different selection.")
    parser.add_argument("--engines", nargs="+", choices=("baseline", "current"), default=["baseline", "current"],
                        help="Select one engine when a full paired comparison is too expensive.")
    parser.add_argument("--baseline-revision", default=BASE, help="Locally available baseline Git commit or ref.")
    parser.add_argument("--current-revision", default=CURRENT, help="Locally available candidate Git commit or ref.")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache" / "ood-transfer-small")
    parser.add_argument("--reuse-cache", type=Path, action="append", default=[],
                        help="Read matching complete rankings from an earlier frozen comparison. May be repeated.")
    parser.add_argument("--reuse-report", type=Path, action="append", default=[],
                        help="Verify the engine identity of legacy cache files against the prior published aggregate.")
    parser.add_argument("--prior-rubq-cache", type=Path, default=ROOT / ".cache" / "rubq-transfer")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "ood_transfer_small.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Aggregate existing runs for these exact frozen inputs.")
    parser.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    bench, cache = args.benchmarks.resolve(), args.cache.resolve()
    if args.worker:
        name, revision = args.worker
        if name not in ALL_NAMES:
            raise ValueError("Unregistered dataset.")
        rank(bench, cache, name, resolve_revision(revision))
        return
    if len(args.datasets) != len(set(args.datasets)) or len(args.engines) != len(set(args.engines)):
        raise ValueError("Select each dataset and engine only once.")
    protocol = freeze(bench, args.datasets, args.engines, baseline_revision=args.baseline_revision,
                      current_revision=args.current_revision)
    freeze_path = cache / "freeze.json"
    cache.mkdir(parents=True, exist_ok=True)
    with FileLock(str(freeze_path) + ".lock"):
        if freeze_path.is_file():
            if json.loads(freeze_path.read_bytes()) != protocol:
                raise ValueError("This cache belongs to another input or engine selection. Choose a new cache directory.")
        else:
            write_json(freeze_path, protocol)
    for source_cache in args.reuse_cache:
        register_reuse(bench, cache, source_cache.resolve(), args.reuse_report, protocol)
    if args.prepare_only:
        return
    if not args.report_only:
        env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                   OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="2")
        for name in args.datasets:
            for engine in ("current", "baseline"):
                if engine not in protocol["engines"]:
                    continue
                rev = protocol["engines"][engine]
                subprocess.run([sys.executable, __file__, "--benchmarks", str(bench), "--cache", str(cache),
                                "--worker", name, rev], check=True, env=env)
    write_json(args.output, report(bench, cache, protocol, args.prior_rubq_cache))


if __name__ == "__main__":
    main()
