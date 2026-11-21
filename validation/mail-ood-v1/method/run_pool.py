"""Run positive and verified-empty mail-search tasks against one project server."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
import sys
from types import SimpleNamespace

def read(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]
def put(path,rows):
    with path.open("w",encoding="utf-8") as out:
        for row in rows:out.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
def summary(rows):
    pos=[r for r in rows if r["state"]=="positive"]
    empty=[r for r in rows if r["state"]=="verified_empty"]
    known=[r for r in pos if r["rr"] is not None]
    def avg(values):
        return round(mean(values),5) if values else None
    return {"queries":len(rows),"positive":len(pos),"verified_empty":len(empty),
            "ndcg_at_10":avg([r["ndcg"] for r in pos]),"known_recall_at_10":avg([r["recall"] for r in pos]),
            "mrr_at_10_single":avg([r["rr"] for r in known]),"success_at_1_single":avg([r["rr"]==1 for r in known]),
            "false_nonempty":sum(r["returned"]>0 for r in empty),"false_nonempty_rate":avg([r["returned"]>0 for r in empty]),
            "false_empty":sum(not r["returned"] for r in pos),"false_empty_rate":avg([not r["returned"] for r in pos]),
            "outside_explicit_window_documents":sum(r["outside_window"] for r in rows)}
def main():
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument("--suite",type=Path,required=True)
    args.add_argument("--project",type=Path,default=Path.cwd(),help="Repository with benchmark.py supporting --queries and --qrels.")
    args.add_argument("--url",default="http://127.0.0.1:8766")
    args.add_argument("--output",type=Path,required=True,help="Directory for this evaluation.")
    args.add_argument("--workers",type=int,default=4)
    args.add_argument("--variants",action="store_true",help="Also include paraphrases and typos, reported separately.")
    args.add_argument("--panel",type=Path,help="Select preassigned primary query IDs from a panel JSONL.")
    a=args.parse_args()
    if a.workers<1:args.error("--workers must be positive")
    sys.path.insert(0,str(a.project.resolve()))
    import benchmark
    if not hasattr(benchmark,"check_service_corpus"):args.error("Use the project benchmark version with corpus verification (e.g. branch search/robust-ranking).")
    path=a.suite.resolve();out=a.output.resolve();out.mkdir(exist_ok=True,parents=True)
    queries=read(path/"queries_dev.jsonl")+read(path/"queries_empty.jsonl")
    labels=read(path/"qrels_dev.jsonl")+read(path/"qrels_empty.jsonl")
    if a.variants:
        queries+=read(path/"queries_variants.jsonl")
        labels+=read(path/"qrels_variants.jsonl")
    if a.panel:
        ids={r["query_id"] for r in read(a.panel)}
        if a.variants:ids.update(r["query_id"] for r in read(path/"cases.jsonl") if r.get("original_query_id") in ids)
        queries=[q for q in queries if q["query_id"] in ids]
        if ids!={q["query_id"] for q in queries}:raise ValueError("Panel and suite differ")
        labels=[r for r in labels if r["query_id"] in ids]
    qids={q["query_id"] for q in queries}
    if len(qids)!=len(queries):raise ValueError("Duplicate query IDs")
    info={r["query_id"]:r for r in read(path/"cases.jsonl") if r["query_id"] in qids}
    if qids!=set(info):raise ValueError("Queries and task metadata differ.")
    # These three files can be passed directly to the project's paired comparison tool.
    put(out/"queries.jsonl",queries);put(out/"qrels.jsonl",labels)
    put(out/"tasks.jsonl",[info[q["query_id"]] for q in queries])
    report=out/"rankings.json"
    benchmark.run(SimpleNamespace(data=path,queries=out/"queries.jsonl",qrels=out/"qrels.jsonl",
                   unjudged=False,url=a.url,k=[1,5,10],repeats=1,concurrency=a.workers,output=report))
    ranked=json.loads(report.read_text())
    dates={d["doc_id"]:d["doc_dt"] for d in read(path/"documents.jsonl")}
    byfamily=defaultdict(list);byvariant=defaultdict(list);bybox=defaultdict(list)
    for q in queries:
        qid=q["query_id"];m=info[qid];b=q["synthetic_user"]
        found=ranked["queries"][qid];ids=found["ranked_ids"];scores=found["scores"].get("10",{})
        targets=set(m.get("target_ids",[]))
        rr=next((1/(p+1) for p,id_ in enumerate(ids) if id_ in targets),0.) if targets else None
        span=m.get("receipt_window")
        outside=sum(dates[i] is None or not span[0]<=dates[i]<=span[1] for i in ids) if span else 0
        entry={"state":m["state"],"rr":rr,"ndcg":scores.get("ndcg"),"recall":scores.get("recall"),"returned":len(ids),"outside_window":outside}
        style=m.get("variant","original")
        byvariant[style].append(entry)
        if style=="original":byfamily[m["family"]].append(entry);bybox[b].append(entry)
    result={"primary":summary(byvariant["original"]),"by_family_primary":{k:summary(v) for k,v in sorted(byfamily.items())},
            "by_mailbox_primary":{k:summary(v) for k,v in sorted(bybox.items())},
            "by_variant":{k:summary(v) for k,v in sorted(byvariant.items())},
            "backend_ms":ranked["timing"]["server_ms"],"scope_check":"All ranked IDs belonged to the requested mailbox and were visible as of query_date.",
            "corpus_sha256":ranked["dataset"]["sha256"]["documents.jsonl"],"settings":{"repeats":1,"workers":a.workers,"top_k":10}}
    (out/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(result["primary"],ensure_ascii=False))
if __name__=="__main__":main()
