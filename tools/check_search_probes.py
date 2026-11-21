"""Run corpus-verified, independently labelled search probes against a local mail API.

    python3 tools/check_search_probes.py --verify-only
    python3 tools/check_search_probes.py --url http://127.0.0.1:8765 --output .cache/probes.json

A completed diagnostic can contain ranking misses. Invalid labels, an incompatible
corpus or a failed HTTP exchange stop the run; partial metrics are not reported.
"""

import argparse
from collections import Counter
from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
LIMIT = 10


def load_corpus(path, expected_hash):
    digest = sha256()
    documents = {}
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            if row["doc_id"] in documents:
                raise ValueError(f"Duplicate document: {row['doc_id']}")
            documents[row["doc_id"]] = row
    if digest.hexdigest() != expected_hash:
        raise ValueError("Corpus and fixture SHA-256 do not match. Supply a fixture labelled for these documents.")
    return documents


def visible(row, case):
    return row["synthetic_user"] == case["mailbox"] and (
        row["doc_dt"] is None or row["doc_dt"] <= case["date"]
    )


def in_window(row, case):
    window = case.get("received")
    return window is None or (row["doc_dt"] is not None and window[0] <= row["doc_dt"] <= window[1])


def verify_labels(suite, documents):
    for entry in suite.get("global_anchors", []):
        literal = entry["literal"].casefold()
        found = {doc_id for doc_id, row in documents.items() if literal in row["doc"].casefold()}
        if found != set(entry["documents"]):
            raise ValueError(f"Global occurrence check failed for {entry['literal']}")
    box_counts = Counter(row["synthetic_user"] for row in documents.values())
    if len({case["id"] for case in suite["cases"]}) != len(suite["cases"]):
        raise ValueError("Repeated probe ID")
    for case in suite["cases"]:
        day = date.fromisoformat(case["date"])
        if case["mailbox"] not in box_counts:
            raise ValueError(f"{case['id']}: unknown mailbox")
        if case.get("received"):
            earliest, latest = map(date.fromisoformat, case["received"])
            if not earliest <= latest <= day:
                raise ValueError(f"{case['id']}: invalid date window")
        patterns = [re.compile(pattern, re.IGNORECASE) for pattern in case["evidence_all"]]
        found = {
            doc_id for doc_id, row in documents.items()
            if visible(row, case) and in_window(row, case) and all(p.search(row["doc"]) for p in patterns)
        }
        if found != set(case["expected"]):
            difference = sorted(found.symmetric_difference(case["expected"]))
            raise ValueError(f"{case['id']}: expected IDs differ from corpus evidence: {difference}")
    for case in suite.get("read_checks", []):
        row = documents[case["doc_id"]]
        if case["status"] != (200 if visible(row, case) else 404):
            raise ValueError(f"Read permissions disagree with labelled status: {case['id']}")
    return dict(box_counts)


def request_json(base, endpoint, body=None):
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = Request(base + endpoint, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, error.read().decode("utf-8", errors="replace")[:300]


def check_search(base, case, documents):
    config = {key: case[key] for key in ("mailbox", "date", "query")}
    status, result = request_json(base, "/api/search", {**config, "limit": LIMIT})
    if status != 200:
        raise ValueError(f"{case['id']}: search returned HTTP {status}: {result}")
    ids = [item["id"] for item in result["items"]]
    if len(ids) > LIMIT or len(ids) != len(set(ids)):
        raise ValueError(f"{case['id']}: response exceeds requested limit or repeats IDs")
    expected = set(case["expected"])
    rank = next((i for i, doc_id in enumerate(ids, 1) if doc_id in expected), None)
    bad_visibility = [doc_id for doc_id in ids if doc_id not in documents or not visible(documents[doc_id], case)]
    bad_window = [doc_id for doc_id in ids if doc_id in documents and not in_window(documents[doc_id], case)]
    return {
        "id": case["id"], "family": case["family"], "expected": case["expected"], "results": ids,
        "first_expected_rank": rank, "invalid_mailbox_or_asof": bad_visibility, "outside_received_window": bad_window,
    }


def check_read(base, case, documents):
    path = "/api/mail/" + quote(case["doc_id"], safe="") + "?" + urlencode(
        {key: case[key] for key in ("mailbox", "date")}
    )
    status, result = request_json(base, path)
    same_document = status == 200 and result.get("id") == case["doc_id"] and result.get("body") == documents[case["doc_id"]]["doc"]
    return {
        "id": case["id"], "status": status, "expected_status": case["status"],
        "ok": status == case["status"] and (status != 200 or same_document),
    }


def summarize(rows, reads):
    positives = [row for row in rows if row["expected"]]
    negatives = [row for row in rows if not row["expected"]]
    return {
        "positive_queries": len(positives),
        "hit_at_1": sum(row["first_expected_rank"] == 1 for row in positives),
        "hit_at_5": sum(row["first_expected_rank"] is not None and row["first_expected_rank"] <= 5 for row in positives),
        "empty_queries": len(negatives),
        "correctly_empty": sum(not row["results"] for row in negatives),
        "returned_items": sum(len(row["results"]) for row in rows),
        "invalid_mailbox_or_asof_items": sum(len(row["invalid_mailbox_or_asof"]) for row in rows),
        "outside_explicit_received_window_items": sum(len(row["outside_received_window"]) for row in rows),
        "read_checks": len(reads), "correct_reads": sum(row["ok"] for row in reads),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765", help="Base URL of the running application.")
    parser.add_argument("--fixture", type=Path, default=ROOT / "tests/fixtures/search_probes.json")
    parser.add_argument("--documents", type=Path, default=ROOT / "data/documents.jsonl")
    parser.add_argument("--output", type=Path, help="Save per-query ranks, returned IDs and scope violations.")
    parser.add_argument("--verify-only", action="store_true", help="Audit all labels locally, without calling the server.")
    args = parser.parse_args()
    fixture_bytes = args.fixture.read_bytes()
    suite = json.loads(fixture_bytes)
    documents = load_corpus(args.documents, suite["documents_sha256"])
    box_counts = verify_labels(suite, documents)
    print(f"Labels verified against {len(documents)} letters: {len(suite['cases'])} search probes, {len(suite['read_checks'])} read checks.")
    if args.verify_only:
        return
    base = args.url.rstrip("/")
    status, data = request_json(base, "/api/mailboxes")
    if status != 200 or {row["id"]: row["count"] for row in data["mailboxes"]} != box_counts:
        raise ValueError("The serving mailbox inventory does not match the fixture corpus.")
    if "corpus_sha256" in data and data["corpus_sha256"] != suite["documents_sha256"]:
        raise ValueError("The server is running against different document bytes.")
    rows = [check_search(base, case, documents) for case in suite["cases"]]
    reads = [check_read(base, case, documents) for case in suite["read_checks"]]
    metrics = summarize(rows, reads)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output:
        output = {
            "name": suite["name"], "tested_at_utc": datetime.now(timezone.utc).isoformat(),
            "url": base, "request_limit": LIMIT, "fixture_sha256": sha256(fixture_bytes).hexdigest(),
            "documents_sha256": suite["documents_sha256"], "metrics": metrics, "cases": rows, "read_checks": reads,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Per-query report: {args.output}")


if __name__ == "__main__":
    main()
