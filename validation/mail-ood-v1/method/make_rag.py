"""Convert the annotated facts to the team's check_rag_facts fixture format."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1]
def jl(path):
    with path.open() as stream:return [json.loads(s) for s in stream if s.strip()]
def patterns(a):
    if a["answerability"]=="unanswerable":return []
    value=a["value"]
    if a["fact"]=="order_total":
        whole,part=value.split(".")
        groups=[]
        while len(whole)>3:
            whole,last=whole[:-3],whole[-3:];groups.insert(0,last)
        return [r"(?<!\d)"+r"[ \u00a0\u202f]?".join([whole]+groups)+r"[.,]"+part+r"(?!\d)",r"(?:\bRUB\b|руб|₽)"]
    if value=="15 октября":return [r"(?:15\s+октябр[ья]|October\s+15(?:th)?)"]
    if value=="350 мм и 600 мм":return [r"(?<!\d)350\s*мм",r"(?<!\d)600\s*мм"]
    if value=="18:13 мск":return [r"\b18[:.]13\b"]
    if value=="9:00":return [r"\b0?9[:.]00\b"]
    if re.fullmatch(r"\d+\.\d+",value):return [r"(?<!\d)"+re.escape(value).replace(r"\.",r"[.,]")+r"(?!\d)"]
    if value.isdigit():return [r"(?<!\d)"+value+r"(?!\d)"]
    return [re.escape(value)]
def main():
    out=ROOT/"rag";out.mkdir(exist_ok=True)
    for p in sorted((ROOT/"suite").glob("*")):
        if not (p/"cases.jsonl").exists():continue
        queries={q["query_id"]:q for q in jl(p/"queries_dev.jsonl")}
        docs={d["doc_id"]:d for d in jl(p/"documents.jsonl")}
        cases=[]
        for m in jl(p/"cases.jsonl"):
            if m.get("variant") or not m.get("answer"):continue
            a=m["answer"];q=queries[m["query_id"]];d=docs[a["doc_id"]];facts=patterns(a)
            if not all(re.search(pat,d["doc"],re.I) for pat in facts):raise ValueError(f"Value absent for {m['query_id']}")
            cases.append({"id":m["query_id"],"mailbox":q["synthetic_user"],"date":q["query_date"],"query":q["query"],
                          "sources":[a["doc_id"]],"facts":facts,"answerability":a["answerability"],"evidence":a["evidence"]})
        suite={"corpus_sha256":json.loads((p/"manifest.json").read_text())["sha256"]["documents.jsonl"],"cases":cases}
        (out/(p.name+".json")).write_text(json.dumps(suite,ensure_ascii=False,indent=2)+"\n")
        print(p.name,len(cases))
if __name__=="__main__":main()
