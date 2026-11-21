"""Test local grounded answers against predeclared facts and source IDs in a fixed corpus."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def get(base, path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(base + path, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def check(base, case, documents):
    scope = {"mailbox": case["mailbox"], "date": case["date"]}
    def visible(doc):
        return doc["synthetic_user"] == scope["mailbox"] and (doc["doc_dt"] is None or doc["doc_dt"] <= scope["date"])
    facts = [re.compile(s, re.I) for s in case.get("facts", [])]
    for doc_id in case["sources"]:
        original = documents[doc_id]
        if not visible(original) or not all(f.search(original["doc"]) for f in facts):
            raise ValueError("Invalid reference evidence for " + case["id"])
    started = perf_counter()
    result = get(base, "/api/answer", {**scope, "query": case["query"]})
    answer_http_ms = round((perf_counter() - started) * 1000, 1)
    opened = []
    for summary in result["sources"]:
        doc_id = summary["id"]
        original = get(base, "/api/mail/" + quote(doc_id, safe="") + "?" + urlencode(scope))
        if not visible(documents[doc_id]) or original["body"] != documents[doc_id]["doc"]:
            raise ValueError("Citation outside the mailbox or different from its original")
        opened.append(doc_id)
    citations = [int(x) for x in re.findall(r"\[(\d+)\]", result["answer"])]
    if any(not 1 <= n <= len(opened) for n in citations):
        raise ValueError("Invalid citation number in " + case["id"])
    evidence = set(case["sources"])
    correct_citations = sum(opened[n - 1] in evidence for n in citations)
    cited_facts = set()
    for line in result["answer"].splitlines():
        refs = [int(x) for x in re.findall(r"\[(\d+)\]", line)]
        if any(opened[n - 1] in evidence for n in refs):
            cited_facts.update(i for i, pattern in enumerate(facts) if pattern.search(line))
    return {"id": case["id"], "answerable": bool(facts), "mode": result["mode"], "answer": result["answer"],
            "answer_http_ms": answer_http_ms, "opened_source_ids": opened,
            "citations": len(citations), "citations_to_expected": correct_citations,
            "facts_expected": len(facts), "facts_cited": len(cited_facts),
            "all_expected_facts_cited": bool(facts) and len(cited_facts) == len(facts) and correct_citations == len(citations)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--documents", type=Path, default=ROOT / "data/documents.jsonl")
    parser.add_argument("--fixture", type=Path, default=ROOT / "tests/fixtures/rag_facts.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    suite = json.loads(args.fixture.read_bytes())
    base = args.url.rstrip("/")
    corpus = sha256(args.documents.read_bytes()).hexdigest()
    if corpus != suite["corpus_sha256"]:
        raise ValueError("The corpus and RAG fixture identify different snapshots")
    with args.documents.open(encoding="utf-8") as file:
        docs = {r["doc_id"]: r for r in map(json.loads, (line for line in file if line.strip()))}
    metadata = get(base, "/api/mailboxes")
    server_hash = metadata.get("corpus_sha256")
    counts = Counter(d["synthetic_user"] for d in docs.values())
    if ((server_hash is not None and server_hash != corpus)
            or counts != {box["id"]: box["count"] for box in metadata["mailboxes"]}):
        raise ValueError("The running service uses a different corpus")
    rows = [check(base, c, docs) for c in suite["cases"]]
    pos = [r for r in rows if r["answerable"]]
    neg = [r for r in rows if not r["answerable"]]
    summary = {"answerable": len(pos), "all_facts_with_expected_citation": sum(r["all_expected_facts_cited"] for r in pos),
               "unwarranted_no_evidence": sum(r["mode"] == "no_evidence" for r in pos),
               "insufficient": len(neg), "explicit_no_evidence": sum(r["mode"] == "no_evidence" for r in neg),
               "citations_on_answerable": sum(r["citations"] for r in pos),
               "expected_source_citations_on_answerable": sum(r["citations_to_expected"] for r in pos),
               "required_facts": sum(r["facts_expected"] for r in pos), "required_facts_with_citation": sum(r["facts_cited"] for r in pos),
               "originals_opened_in_scope": sum(len(r["opened_source_ids"]) for r in rows),
               "modes": dict(Counter(r["mode"] for r in rows))}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        report = {"corpus_sha256": corpus, "server_corpus_check": "sha256" if server_hash else "mailbox_counts_only",
                  "fixture_sha256": sha256(args.fixture.read_bytes()).hexdigest(),
                  "scope":"Small task-authored regression set with deterministic, predeclared facts. References must open in the same snapshot. Fact-pattern and source checks do not evaluate arbitrary generated claims.",
                  "summary":summary,"cases":rows}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
