"""Check both halves of the publisher's independent VK event and date diagnostics.

Use --pool with the local OOD package and pass two VK API endpoints in revision
order. Select --all-mailboxes to include the entire supplied diagnostic set.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark import check_service_corpus, fetch, load_benchmark
from check_rubq import read_lines, write_json
from check_mail_ood import AFTER, BEFORE, check_inputs, commit_id, digest


def development_pairs(pool, all_mailboxes=False):
    release = pool / "suite" / "vk_dev"
    diagnostic = pool / "diagnostics"
    check_inputs(pool, "vk_dev")
    docs, queries, labels, source = load_benchmark(
        release, diagnostic / "queries.jsonl", diagnostic / "qrels.jsonl"
    )
    boxes = {q["synthetic_user"] for q in read_lines(release / "queries_dev.jsonl")}
    if all_mailboxes:
        check_inputs(pool, "vk_holdout")
        boxes.update(q["synthetic_user"] for q in read_lines(pool / "suite/vk_holdout/queries_dev.jsonl"))
    by_id = {q["query_id"]: q for q in queries}
    pairs = defaultdict(dict)
    seen = set()
    for row in read_lines(diagnostic / "pairs.jsonl"):
        qid, family, pair, state = (row[k] for k in ("query_id", "family", "pair", "state"))
        query = by_id[qid]
        target = row["source_document"]
        if qid in seen or state in pairs[family, pair] or state not in {"positive", "verified_empty"}:
            raise ValueError("A diagnostic case or one of its states is duplicated or unknown.")
        seen.add(qid)
        known = {did for did, grade in labels[qid].items() if grade > 0}
        if known != ({target} if state == "positive" else set()):
            raise ValueError("The pair metadata and supplied judgments disagree.")
        pairs[family, pair][state] = (query, target)
    if seen != set(by_id):
        raise ValueError("Missing diagnostic metadata.")
    eligible, reserved = defaultdict(list), Counter()
    for (family, _), states in pairs.items():
        positive, target = states["positive"]
        empty, same_target = states["verified_empty"]
        owner = positive["synthetic_user"]
        if (owner != empty["synthetic_user"] or positive["query_date"] != empty["query_date"]
                or target != same_target or docs[target][0] != owner):
            raise ValueError("The two members of a pair use different scopes or anchors.")
        if owner in boxes:
            eligible[family].append((positive, empty, target))
        else:
            reserved[family] += 1
    source["sha256"]["pairs"] = digest(diagnostic / "pairs.jsonl")
    return docs, eligible, reserved, source["sha256"]


def measure(url, docs, corpus_sha, families):
    service_check = check_service_corpus(url, docs, corpus_sha)
    detail, totals = {}, {}
    for family, cases in families.items():
        rows = []
        for positive, empty, target in cases:
            replies = []
            for q in (positive, empty):
                ids, _, _ = fetch(url, q, 10)
                if len(ids) != len(set(ids)) or len(ids) > 10:
                    raise ValueError("Duplicate result or invalid ranking depth.")
                for did in ids:
                    owner, day = docs[did]
                    if owner != q["synthetic_user"] or day and day > q["query_date"]:
                        raise ValueError("A reply contains a document outside the mailbox or query date.")
                replies.append(ids)
            hit, abstained = target in replies[0], not replies[1]
            rows.append((hit, abstained))
        detail[family] = [hit and abstained for hit, abstained in rows]
        totals[family] = {
            "pairs": len(rows),
            "positive_hit_at_10": sum(hit for hit, _ in rows),
            "correct_empty": sum(abstained for _, abstained in rows),
            "both_conditions_correct": sum(detail[family]),
        }
    return {"service_corpus_check": service_check, "families": totals}, detail


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--all-mailboxes", action="store_true")
    p.add_argument("--baseline-url", default="http://127.0.0.1:18773")
    p.add_argument("--current-url", default="http://127.0.0.1:8766")
    p.add_argument("--baseline-revision", default=BEFORE)
    p.add_argument("--current-revision", default=AFTER)
    p.add_argument("--output", type=Path, default=ROOT / "results" / "ood_event_pairs.json")
    args = p.parse_args()
    docs, families, reserved, hashes = development_pairs(args.pool, args.all_mailboxes)
    if args.all_mailboxes and reserved:
        raise ValueError("Some diagnostic mailboxes are missing from the supplied suites.")
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [executor.submit(measure, url, docs, hashes["documents.jsonl"], families)
                for url in (args.baseline_url, args.current_url)]
        (baseline, before), (current, after) = [job.result() for job in jobs]
    changes = {
        family: {"improved": sum(not a and b for a, b in zip(before[family], after[family], strict=True)),
                 "regressed": sum(a and not b for a, b in zip(before[family], after[family], strict=True))}
        for family in families
    }
    write_json(args.output, {
        "engines": {"baseline": commit_id(args.baseline_revision), "current": commit_id(args.current_revision)},
        "sha256": hashes, "cutoff": 10,
        "selection": ("Both halves of every supplied diagnostic pair." if args.all_mailboxes else
                      "Both halves of every supplied diagnostic pair whose mailbox is in vk_dev."),
        "reserved_pairs_by_family": dict(reserved), "baseline": baseline, "current": current,
        "paired_changes_both_conditions": changes,
    })


if __name__ == "__main__":
    main()
