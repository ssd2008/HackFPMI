from pathlib import Path
import hashlib
import json
import shutil

root = Path.cwd()
work = root / ".cache/iteration"
def read(path):
    return json.loads(path.read_text())
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def snapshot(key):
    return {v:read(work / f"r2-{v}-{key}.json") for v in ("before","after")}
def copy_run(key, filename):
    shutil.copyfile(work / f"r2-after-{key}.json", root / "results" / filename)
def quality(key):
    versions = snapshot(key)
    old, new = [versions[k]["queries"] for k in ("before","after")]
    moves = {"better":0,"worse":0,"tie":0}
    for qid,row in new.items():
        b,a = old[qid]["scores"]["10"]["ndcg"],row["scores"]["10"]["ndcg"]
        outcome = "better" if a>b+1e-9 else ("worse" if a<b-1e-9 else "tie")
        moves[outcome]+=1
    return {"queries":len(new), "nDCG10_query_changes":moves,"quality":{v:d["quality_first_pass"] for v,d in versions.items()},
            "quality_queries":versions['after']['quality_queries'],"single_positive_queries":versions['after']['single_relevant_queries'],
            "max_macro_recall":versions['after']['recall_upper_bound']}
def timing(key):
    return {v:d["timing"] for v,d in snapshot(key).items()}
raw_ablation = read(root / ".cache/ranking-study/ablation-summary.json")
spell = read(root / ".cache/spelling-study/report.json")
checks = {
 "first_probe_set":{
   "fixture_sha256":digest(root/"tests/fixtures/search_probes.json"),
   "initial_mvp":read(root/".cache/search-probes-baseline.json")["metrics"],
   "first_unseen_pass":read(work/"first-frozen-probes.json")["metrics"],
   "retest_after_fix":read(root/"results/search_probes.json")["metrics"]},
 "second_probe_set":{
   "fixture_sha256":digest(root/"tests/fixtures/search_probes_extra.json"),
   "initial_mvp":read(work/"extra-baseline.json")["metrics"],
   "first_unseen_pass_final":read(root/"results/search_probes_extra.json")["metrics"]}}
data={
 "baseline_app_commit":"5fe3f5d3739ac523d0b4088f15eef4b50391dd28",
 "candidate_source_sha256":{p:digest(root/"mailsearch"/p) for p in ("retrieval.py","dates.py","dense.py","spelling.py")},
 "document_sha256":digest(root/"data/documents.jsonl"),
 "benchmark_inputs_sha256":{p:digest(root/"tests/fixtures"/p) for p in ("queries_no_date.jsonl","queries_paraphrased.jsonl","queries_typos.jsonl")},
 "protocol":"Tests used the real search API, identical data, top-k, query order and client count for both versions. Final pass on an Apple M5 Pro, 48 GiB. No query vector or result cache. Top-50 used for quality, top-10 with five repetitions for load.",
 "quality":{"public_dev":quality("wide"),"known_paraphrases":quality("para"),"synthetic_typos":quality("typo")},
 "speed":{"public_1_client":timing("one"),"public_8_clients":timing("eight"),"receipt_filter_removed_8_clients":timing("open")},
 "probes":checks,
 "development_experiments":{
   "component_scores_ndcg10":{variant:{name:summary["quality"]["10"]["nDCG"] for name,summary in row["sets"].items()} for variant,row in raw_ablation["variants"].items()},
   "cross_encoder":{"model":"cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "revision":"1427fd652930e4ba29e8149678df786c240d8825", "max_candidates":32,"ndcg10":{"e5":{"dev":0.93515,"paraphrases":0.88246},"e5_plus_reranker":{"dev":0.95533,"paraphrases":0.90649},"original_mvp_plus_reranker":{"dev":0.95877,"paraphrases":0.80044}},"rerank_only_p99_ms":{"cpu_fp32":470.87,"mps_fp16":84.88}},
   "spelling_ablation":{"threshold_min_letters":6,"queries_changed":spell["by_group"]["typos"]["corrected_queries"],
                        "ndcg10":{k:v["10"]["ndcg"] for k,v in spell["by_group"]["typos"]["quality"].items()},
                        "standalone_warm_ms":spell["warm_normalize_ms"]}
 }}
(root/"results/search_comparison.json").write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
copy_run("one","public_dev.json")
copy_run("eight","public_dev_8_clients.json")
