"""Run the two open mail OOD RAG fixtures over loopback, caching private rows locally."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

from check_rag_facts import check, get

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / ".cache/ood-mail"
ORIGINAL_REPORT = ROOT / "results/ood_mail_rag.json"
DEFAULT_VERSIONS = {"before": "5fe3f5d", "after": "b37c309"}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read_lines(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_inputs(pool, source):
    split = source + "_dev"
    fixture = pool / "rag" / (split + ".json")
    panel_path = pool / "panels" / (split + ".jsonl")
    folder = pool / "suite" / split
    docs_path = folder / "documents.jsonl"
    definition = json.loads(fixture.read_text())
    assert definition["corpus_sha256"] == digest(docs_path)
    docs = {r["doc_id"]: r for r in read_lines(docs_path)}
    cases = {r["id"]: r for r in definition["cases"]}
    panel = [r["query_id"] for r in read_lines(panel_path) if r["query_id"] in cases]
    metadata = {r["query_id"]: r for r in read_lines(folder / "cases.jsonl") if r["query_id"] in cases}
    assert len(panel) == len(set(panel)) and cases.keys() == metadata.keys()
    for cid, c in cases.items():
        assert c["answerability"] in ("answerable", "unanswerable")
        assert bool(c["facts"]) == (c["answerability"] == "answerable")
        assert c["sources"] == [metadata[cid]["answer"]["doc_id"]]
    return {"cases": cases, "panel": panel, "metadata": metadata, "docs": docs,
            "hashes": {"fixture": digest(fixture), "panel": digest(panel_path), "documents": digest(docs_path)}}


def server_check(url, data):
    if urlsplit(url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("The mail OOD requests must stay on loopback.")
    server = get(url, "/api/mailboxes")
    counts = Counter(d["synthetic_user"] for d in data["docs"].values())
    if counts != {r["id"]: r["count"] for r in server["mailboxes"]}:
        raise ValueError("Server mailbox sizes differ from the fixture corpus.")
    if server.get("corpus_sha256") is None:
        return "mailbox_counts"
    if server["corpus_sha256"] != data["hashes"]["documents"]:
        raise ValueError("Server corpus digest differs from the fixture.")
    return "sha256"


def acquire(pool, urls, phase, cache, versions):
    inputs = {s: load_inputs(pool, s) for s in ("vk", "personal")}
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    checks = {(s, v): server_check(urls[s, v], inputs[s]) for s in inputs for v in versions}
    for s, data in inputs.items():
        for v in versions:
            folder = cache / s / v
            folder.mkdir(mode=0o700, parents=True, exist_ok=True)
            stamp = {"version": versions[v], **data["hashes"], "server_corpus_check": checks[s, v]}
            path = folder / "input.json"
            if path.exists() and json.loads(path.read_text()) != stamp:
                raise ValueError("Cached run uses a different frozen input or revision.")
            path.write_text(json.dumps(stamp) + "\n")
    for current_phase in (("panel", "remaining") if phase == "all" else (phase,)):
        # Personal has a single case. Finish that independent stratum first.
        for source in ("personal", "vk"):
            data = inputs[source]
            ids = data["panel"] if current_phase == "panel" else [i for i in data["cases"] if i not in set(data["panel"])]
            for n, cid in enumerate(ids, 1):
                for version in versions:
                    # Opaque filenames keep original query/document IDs inside the private records.
                    name = sha256(cid.encode()).hexdigest() + ".json"
                    target = cache / source / version / name
                    if target.exists():
                        continue
                    result = check(urls[source, version], data["cases"][cid], data["docs"])
                    tmp = target.with_suffix(".tmp")
                    tmp.write_text(json.dumps(result, ensure_ascii=False) + "\n")
                    os.replace(tmp, target)
                if n % 10 == 0 or n == len(ids):
                    print(f"{source} {current_phase}: {n}/{len(ids)} paired requests complete", flush=True)


REF = re.compile(r"\[(\d+)\]")
ORDER_ID = re.compile(r"\bsyn-[a-z0-9]+\b", re.I)
NUM = re.compile(r"\d+(?:[.,:/-]\d+)*")
CURRENCY = re.compile(r"(?<!\w)(?:rub|руб(?:ль|ля|лей|лями|лях)?)(?!\w)|₽", re.I)
MONEY_VALUE = re.compile(
    r"(?<![\w.,])(\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[.,](\d+))?"
    r"\s*(?:rub\b|руб(?:ль|ля|лей|лями|лях)?\b|₽)", re.I)
TRACK = re.compile(r"трек[ -]?(?:номер|код)|tracking|почтовый идентификатор", re.I)
ABSENT = re.compile(r"недостаточно|нет (?:сведений|данных|информации)|не (?:указан|найден|содерж|привед)", re.I)


def normalized(text):
    return " ".join(text.casefold().replace("ё", "е").split())


def figures(text):
    text = ORDER_ID.sub(" ", text)
    text = re.sub(r"(?<=\d)[ \u00a0\u202f](?=\d{3}(?:\D|$))", "", text)
    return {s.replace(",", ".") for s in NUM.findall(text)}


def verifiable_line(line, original, mode):
    text = normalized(REF.sub("", line)).rstrip(". ")
    body = normalized(original)
    if text.startswith("«") and text.endswith("»"):
        pieces = text[1:-1].split("» «")
        return all(piece.rstrip("… ") in body for piece in pieces)
    if text in body:
        return True
    if mode != "generative" or not figures(text) <= figures(body):
        return False
    if not set(ORDER_ID.findall(text)) <= set(ORDER_ID.findall(body)):
        return False
    if CURRENCY.search(text) and not CURRENCY.search(body):
        return False
    # Restricted to these fixtures' stated quantities, durations and identifiers.
    # Unexpected generated relations are flagged for inspection, never accepted
    # just because the expected numeral and citation occur somewhere in the reply.
    units_removed = NUM.sub(" ", ORDER_ID.sub(" ", CURRENCY.sub(" ", text)))
    words = re.findall(r"[^\W\d_]+", units_removed)
    allowed = re.compile(r"на|до|в|за|и|от|по|был[аио]?|состав(?:ила?|ляет)|заказ\w*|сумм\w*|общ\w*|итого|стоил\w*|стоимост\w*|цен[ауы]|мск|мм")
    return all(allowed.fullmatch(word) for word in words)


def grade(row, case, metadata, docs):
    refs = row["opened_source_ids"]
    expected = set(case["sources"])
    source = docs[case["sources"][0]]["doc"]
    if normalized(case["evidence"]) not in normalized(source):
        raise ValueError("The author's evidence fragment is absent from the original.")
    positive = case["answerability"] == "answerable"
    patterns = [re.compile(p, re.I) for p in case["facts"]]
    gold = metadata["answer"]
    money = gold["fact"] == "order_total" or bool(re.fullmatch(r"\d+[.]\d+", str(gold.get("value", ""))))
    if not positive and TRACK.search(source):
        raise ValueError("A source labeled unanswerable mentions a tracking identifier; inspect the source.")
    proven, requested, cited = 0, 0, 0
    found = set()
    conflict = False
    whole_amount_in_one_claim = False
    unknown = False
    for line in row["answer"].splitlines():
        markers = [int(v) for v in REF.findall(line)]
        for n in markers:
            did = refs[n-1]
            good = verifiable_line(line, docs[did]["doc"], row["mode"])
            cited += 1
            proven += good
            if not good:
                unknown = True
            if not positive:
                continue
            # A number may coincide in different orders. Credit the requested relation
            # only in its registered source, with any stated order identity intact.
            right_ids = set(ORDER_ID.findall(normalized(case["query"])))
            line_ids = set(ORDER_ID.findall(normalized(line)))
            wrong_order = bool(right_ids and line_ids - right_ids)
            matched = {i for i,p in enumerate(patterns) if p.search(line)}
            if did in expected and good and not wrong_order and not ABSENT.search(line):
                found.update(matched)
                requested += bool(matched)
                if len(matched) == len(patterns):
                    whole_amount_in_one_claim = True
            if money:
                for m in MONEY_VALUE.finditer(line):
                    stated = re.sub(r"\s", "", m[1]) + ("." + m[2] if m[2] else "")
                    if not patterns[0].search(stated):
                        conflict = True
    abstains = row["mode"] == "no_evidence" and not refs and bool(ABSENT.search(row["answer"]))
    if not positive:
        answer = normalized(REF.sub("", row["answer"]))
        if abstains or (TRACK.search(answer) and ABSENT.search(answer) and not figures(answer)):
            outcome = "justified_abstention"
        elif TRACK.search(answer) or (ORDER_ID.fullmatch(answer.strip("«». ")) and not ABSENT.search(answer)):
            outcome = "unsupported_requested_fact"
        else:
            outcome = "nonanswer_excerpt"
    elif abstains:
        outcome = "unwarranted_abstention"
    elif unknown:
        outcome = "requires_review"
    elif len(found) == len(patterns) and (not money or whole_amount_in_one_claim):
        if conflict:
            outcome = "correct_fact_plus_conflicting_answer"
        elif any(did not in expected for did in refs) or requested != cited:
            outcome = "correct_fact_plus_off_topic_excerpt"
        else:
            outcome = "complete_grounded_answer"
    elif conflict:
        outcome = "incorrect_requested_fact"
    else:
        outcome = "partial_or_nonanswer"
    return {"positive": positive, "class": outcome, "family": metadata["family"],
            "mode": row["mode"], "cited": cited, "opened": len(refs), "supported_citations": proven,
            "expected_citations": row["citations_to_expected"], "fact_units": len(patterns),
            "cited_fact_units": len(found), "answer_ms": row["answer_http_ms"],
            "answer_digest": sha256(row["answer"].encode()).digest(), "reference_ids": refs}


def aggregate(rows):
    positives = [r for r in rows if r["positive"]]
    negatives = [r for r in rows if not r["positive"]]
    classes = Counter(r["class"] for r in rows)
    return {
        "total": len(rows), "answerable": len(positives), "insufficient": len(negatives),
        "full_grounded_answers": classes["complete_grounded_answer"],
        "all_requested_fields_cited_even_with_extraneous_material": sum(
            r["cited_fact_units"] == r["fact_units"] for r in positives),
        "unwarranted_abstentions": classes["unwarranted_abstention"],
        "justified_abstentions": classes["justified_abstention"],
        "irrelevant_replies_on_insufficient": classes["nonanswer_excerpt"],
        "unsupported_requested_values_on_insufficient": classes["unsupported_requested_fact"],
        "counts_by_outcome": dict(sorted(classes.items())),
        "modes": dict(sorted(Counter(r["mode"] for r in rows).items())),
        "required_fact_units_on_positives": sum(r["fact_units"] for r in positives),
        "required_fact_units_cited_on_positives": sum(r["cited_fact_units"] for r in positives),
        "positive_citations": sum(r["cited"] for r in positives),
        "positive_citations_to_registered_sources": sum(r["expected_citations"] for r in positives),
        "all_cited_lines": sum(r["cited"] for r in rows),
        "all_cited_lines_supported_by_own_original": sum(r["supported_citations"] for r in rows),
        "original_links_opened_and_body_scope_verified": sum(r["opened"] for r in rows),
    }


def response_changes(ids, parsed, cases):
    changes = Counter()
    transitions = Counter()
    for cid in ids:
        prior, current = (parsed[v][cid] for v in ("before", "after"))
        old_refs, new_refs = prior["reference_ids"], current["reference_ids"]
        added = set(new_refs) - set(old_refs)
        removed = set(old_refs) - set(new_refs)
        expected = set(cases[cid]["sources"])
        text_changed = prior["answer_digest"] != current["answer_digest"]
        changes["changed_answer_text"] += text_changed
        changes["changed_ordered_sources"] += old_refs != new_refs
        changes["changes_in_source_order_only"] += old_refs != new_refs and not (added or removed)
        changes["changed_text_with_identical_ordered_sources"] += text_changed and old_refs == new_refs
        changes["added_source_references"] += len(added)
        changes["removed_source_references"] += len(removed)
        changes["added_references_outside_registered_sources"] += len(added - expected)
        changes["removed_references_outside_registered_sources"] += len(removed - expected)
        changes["changed_mode"] += prior["mode"] != current["mode"]
        if prior["class"] != current["class"]:
            transitions[prior["class"] + " -> " + current["class"]] += 1
    return {"compared": len(ids), **changes, "outcome_transitions": dict(sorted(transitions.items()))}


def reply_signature(row):
    return (row["mode"], row["answer"], tuple(row["opened_source_ids"]))


def control_changes(pool, urls, cache, versions, note):
    """Repeat only previously changed responses, sequentially in opposite version orders."""
    location = cache / "load-control"
    location.mkdir(mode=0o700, exist_ok=True)
    overall = {"versions": versions, "protocol": "Both versions' exact top five are fetched for every changed first-pass response. Each changed case is re-run twice per version in a single client. The second repetition reverses version order. The original primary measurements are preserved.", "operational_note": note, "sources": {}}
    for source in ("vk", "personal"):
        data = load_inputs(pool, source)
        for v in versions:
            server_check(urls[source, v], data)
        folder = location / source
        folder.mkdir(mode=0o700, exist_ok=True)
        records = []
        for cid, case in data["cases"].items():
            name = sha256(cid.encode()).hexdigest() + ".json"
            original = {v: json.loads((cache / source / v / name).read_text()) for v in versions}
            if reply_signature(original["before"]) == reply_signature(original["after"]):
                continue
            record_path = folder / name
            if record_path.exists():
                record = json.loads(record_path.read_text())
            else:
                payload = {"query": case["query"], "mailbox": case["mailbox"], "date": case["date"], "limit": 5}
                tops = {v: [row["id"] for row in get(urls[source,v], "/api/search", payload)["items"]] for v in versions}
                repeats = []
                for order in (("before", "after"), ("after", "before")):
                    repeats.append({v: check(urls[source,v], case, data["docs"]) for v in order})
                record = {"versions": versions, "query_id": cid, "ranked_ids_top5": tops, "repeats": repeats}
                tmp = record_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(record, ensure_ascii=False) + "\n")
                os.replace(tmp, record_path)
            assert record["versions"] == versions and record["query_id"] == cid
            repeats = record["repeats"]
            check_rows = []
            for run in repeats:
                check_rows.append({v: grade(row, case, data["metadata"][cid], data["docs"]) for v,row in run.items()})
            records.append((original, record["ranked_ids_top5"], repeats, check_rows))
        summary = {"changed_initial_cases": len(records), "rerun_count_per_version": 2 * len(records),
                   "identical_ordered_search_top5": sum(t['before']==t['after'] for _,t,_,_ in records),
                   "pairwise_identical_rerun_text_sources_and_mode": sum(
                       reply_signature(r['before'])==reply_signature(r['after']) for _,_,rs,_ in records for r in rs),
                   "pairwise_identical_rerun_quality": sum(
                       r['before']['class']==r['after']['class'] for _,_,_,judgments in records for r in judgments)}
        for v in versions:
            durations = [orig[v]["answer_http_ms"] for orig,_,_,_ in records]
            summary[v] = {"stable_across_both_repetitions": sum(
                              reply_signature(rs[0][v])==reply_signature(rs[1][v]) for _,_,rs,_ in records),
                          "reruns_equal_own_initial_response": sum(
                              reply_signature(r[v])==reply_signature(orig[v]) for orig,_,rs,_ in records for r in rs),
                          "initial_http_ms_min_max": [min(durations),max(durations)] if durations else None,
                          "rerun_outcomes": dict(Counter(r[v]['class'] for _,_,_,j in records for r in j))}
        overall["sources"][source] = summary
    (location / "summary.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2) + "\n")


def evaluation(pool, output, cache, versions):
    result = {"status": "complete", "versions": versions, "method": {
        "scope": "Original open RAG fixtures only. The authored retrieval panels are reported by intersection with these fixtures. All remaining open cases are evaluated once in the full section. Panels are subsets of the full rows.",
        "comparison": "Each query is posted unchanged to both real /api/answer endpoints with its own mailbox and as-of date. One request at a time, original response cache, two pinned local model services. Baseline reports mailbox counts, candidate confirms the corpus digest.",
        "assessment": "Rule- and agent-adjudicated, no independent human raters. A complete answer supplies every requested value or field in a responsive claim attributed to the registered original, without mixing in a conflicting result or another event. Each citation is opened with the request scope; the body is compared with the local corpus. Templates and the small set of natural questions were inspected separately.",
        "unanswerable": "The referenced order letter is present but omits the requested tracking number. Reporting another number as tracking is distinct from a real excerpt about the order that fails to answer.",
        "privacy": "Published aggregates only. Raw responses, source IDs and resumed inputs stay in the private local cache."
    }, "sources": {}}
    for source in ("vk", "personal"):
        data = load_inputs(pool, source)
        parsed = {}
        corpus_checks = {}
        for v in versions:
            parsed[v] = {}
            folder = cache / source / v
            control = json.loads((folder / "input.json").read_text())
            corpus_checks[v] = control["server_corpus_check"]
            if control["version"] != versions[v] or any(control[k] != h for k, h in data["hashes"].items()):
                raise ValueError("The response cache and evaluation inputs differ.")
            for cid, case in data["cases"].items():
                path = folder / (sha256(cid.encode()).hexdigest() + ".json")
                if path.exists():
                    row = json.loads(path.read_text())
                    assert row["id"] == cid
                    parsed[v][cid] = grade(row, case, data["metadata"][cid], data["docs"])
        subsets = {}
        for key, selected in (("author_panel", data["panel"]), ("full_open", list(data["cases"]))):
            common = [i for i in selected if all(i in parsed[v] for v in versions)]
            if len(common) != len(selected):
                result["status"] = "in_progress"
            section = {"defined_rows": len(selected), "paired_rows": len(common)}
            for v in versions:
                rows = [parsed[v][i] for i in common]
                section[v] = {"all": aggregate(rows), "by_family": {
                    f: aggregate([r for r in rows if r["family"] == f]) for f in sorted({r["family"] for r in rows})}}
            positive = [i for i in common if parsed["before"][i]["positive"]]
            deltas = [int(parsed["after"][i]["class"] == "complete_grounded_answer") -
                      int(parsed["before"][i]["class"] == "complete_grounded_answer") for i in positive]
            section["paired_answerable"] = {"candidate_wins": deltas.count(1), "ties": deltas.count(0), "candidate_losses": deltas.count(-1)}
            if versions != DEFAULT_VERSIONS:
                section["response_changes"] = response_changes(common, parsed, data["cases"])
            subsets[key] = section
        result["sources"][source] = {"sha256": data["hashes"], **subsets}
        if versions != DEFAULT_VERSIONS:
            result["sources"][source]["server_corpus_check"] = corpus_checks
            result["method"]["comparison"] = (
                "Each query is posted unchanged to both local /api/answer endpoints with its own mailbox and as-of date. "
                "Revision labels identify the pinned deployments. The response metadata verifies corpus identity.")
    receipt = cache / "baseline-reuse.json"
    if receipt.is_file() and versions != DEFAULT_VERSIONS:
        reuse = json.loads(receipt.read_text())
        if reuse["baseline_revision"] != versions["before"]:
            raise ValueError("The reused baseline does not match the requested revision.")
        counts = {}
        for source, proof in reuse["sources"].items():
            files = sorted(p for p in (cache / source / "before").glob("*.json") if p.name != "input.json")
            digest = sha256(b"".join(sha256(path.read_bytes()).digest() for path in files)).hexdigest()
            if len(files) != proof["reused_frozen_response_count"] or digest != proof["sha256_of_ordered_response_hashes"]:
                raise ValueError("Saved baseline responses differ from the frozen-copy receipt.")
            counts[source] = len(files)
        result["method"]["reused_baseline_api_responses"] = counts
    if result["status"] == "complete" and any(
        cohort["full_open"][v]["all"]["counts_by_outcome"].get("requires_review", 0)
        for cohort in result["sources"].values() for v in versions
    ):
        result["status"] = "needs_manual_review"
    control_path = cache / "load-control/summary.json"
    if control_path.exists():
        control = json.loads(control_path.read_text())
        if control["versions"] != versions:
            raise ValueError("The separate load-control round belongs to another revision pair.")
        result["load_control"] = control
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--phase", choices=("panel", "remaining", "all"), default="all")
    p.add_argument("--score-only", action="store_true", help="Recompute aggregates from saved local responses.")
    p.add_argument("--control-changes", action="store_true",
                   help="After the full pass, rerun every changed case twice on both endpoints, sequentially.")
    p.add_argument("--control-note", default="", help="Operational circumstances of the first and control passes.")
    p.add_argument("--baseline-revision", default=DEFAULT_VERSIONS["before"])
    p.add_argument("--candidate-revision", default=DEFAULT_VERSIONS["after"])
    p.add_argument("--cache", type=Path, help="Private run directory inside .cache/ood-mail.")
    p.add_argument("--output", type=Path, help="Public aggregate file. New revision pairs use distinct defaults.")
    for source, ports in {"vk": (18773, 8766), "personal": (18777, 18776)}.items():
        for version, port in zip(DEFAULT_VERSIONS, ports):
            p.add_argument(f"--{source}-{version}", default=f"http://127.0.0.1:{port}")
    a = p.parse_args()
    versions = {"before": a.baseline_revision, "after": a.candidate_revision}
    if not all(label.strip() for label in versions.values()):
        p.error("Both revision identifiers must be nonempty.")
    original = versions == DEFAULT_VERSIONS
    signature = sha256(json.dumps(versions, sort_keys=True).encode()).hexdigest()[:12]
    cache = (a.cache or (PRIVATE / ("rag" if original else f"rag-{signature}"))).resolve()
    if cache.is_relative_to(ROOT) and not cache.is_relative_to((ROOT / ".cache").resolve()):
        p.error("For a cache inside this repository, choose the ignored .cache directory for raw responses.")
    output = a.output or (ORIGINAL_REPORT if original else ROOT / f"results/ood_mail_rag_{signature}.json")
    if not original and (cache == (PRIVATE / "rag").resolve() or output.resolve() == ORIGINAL_REPORT.resolve()):
        p.error("Choose fresh paths for this pair; the original evaluation is frozen.")
    urls = {(s, v): getattr(a, s + "_" + v).rstrip("/") for s in ("vk", "personal") for v in versions}
    os.umask(0o077)
    if not a.score_only:
        acquire(a.pool, urls, a.phase, cache, versions)
    if a.control_changes:
        control_changes(a.pool, urls, cache, versions, a.control_note)
    evaluation(a.pool, output, cache, versions)


if __name__ == "__main__":
    main()
