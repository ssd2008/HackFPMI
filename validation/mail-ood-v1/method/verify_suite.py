"""Verify checksums, exact-rule qrels, review coverage, split separation and cited evidence."""
from __future__ import annotations
from collections import Counter,defaultdict
from email.utils import parseaddr
import hashlib
import json
from pathlib import Path
import re
import unicodedata

BASE=Path(__file__).resolve().parents[1]
def read(p):
    with p.open(encoding="utf-8") as f:return [json.loads(s) for s in f if s.strip()]
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def norm(s):return " ".join(unicodedata.normalize("NFKC",s).casefold().replace("ё","е").split())
def require(ok,why):
    if not ok:raise ValueError(why)
def index(path):
    raw=read(path);docs={d["doc_id"]:d for d in raw};require(len(raw)==len(docs),"duplicate documents")
    domains,titles,entities=defaultdict(set),defaultdict(set),defaultdict(set)
    for did,d in docs.items():
        box=d["synthetic_user"];text=d["doc"]
        header=re.split(r"(?:^|\n)[ \t]*текст письма\s*:",text,1,flags=re.I)[0][:2000]
        sender=re.search(r"^\s*(?:от|from|отправитель)\s*:[ \t]*(.*)$",header,re.I|re.M)
        if sender:
            a=parseaddr(sender[1])[1].casefold()
            if "@" in a:domains[box,a.rsplit("@",1)[-1].rstrip("> ")].add(did)
        sub=re.search(r"^\s*(?:тема|subject)\s*:[ \t]*(.*)$",header,re.I|re.M)
        if sub:titles[box,norm(sub[1])].add(did)
        for token in set(re.findall(r"(?<![\w-])SYN-[a-f0-9]{6,}(?![\w-])",text,re.I)):entities[box,token.upper()].add(did)
    return docs,domains,titles,entities
def check(folder,packed,prior):
    docs,domains,titles,entities=packed;manifest=json.loads((folder/"manifest.json").read_text())
    require(all(sha(folder/k)==v for k,v in manifest["sha256"].items()),"checksum "+folder.name)
    pos=read(folder/"queries_dev.jsonl");empty=read(folder/"queries_empty.jsonl");variants=read(folder/"queries_variants.jsonl")
    q={x["query_id"]:x for x in pos+empty+variants}
    require(len(q)==len(pos)+len(empty)+len(variants),"duplicate queries")
    c={x["query_id"]:x for x in read(folder/"cases.jsonl")};require(q.keys()==c.keys(),"metadata IDs")
    require(len(pos)==manifest["queries"] and len(docs)==manifest["documents"],"manifest counts")
    for row in q.values():require(set(row)=={"query_id","synthetic_user","query","query_date"},"query fields")
    labels=defaultdict(dict);stats=Counter()
    for name,allowed in (("qrels_dev",pos),("qrels_empty",empty),("qrels_variants",variants)):
        ids={v["query_id"] for v in allowed}
        links=read(folder/(name+".jsonl"))
        if name=="qrels_dev":require(len(links)==manifest["qrels"],"qrel count")
        for row in links:
            k,i,g=row["query_id"],row["doc_id"],row["relevance"]
            require(set(row)=={"query_id","doc_id","relevance"} and k in ids and i in docs and i not in labels[k] and g in {0,1,2},"qrel reference")
            d=docs[i];query=q[k]
            require(d["synthetic_user"]==query["synthetic_user"] and (not d["doc_dt"] or d["doc_dt"]<=query["query_date"]),"access or future")
            labels[k][i]=g
    textset=set();judged=set();groups=set();mailboxes=set();primaries=[]
    for k,query in q.items():
        m=c[k];box=query["synthetic_user"];at=query["query_date"];span=m.get("receipt_window")
        relevant={i for i,g in labels[k].items() if g>0};direct={i for i,g in labels[k].items() if g==2}
        require(bool(relevant)==(m["state"]=="positive") and len(direct)==m["relevant_count"],"state/count")
        if span:require(all(docs[i]["doc_dt"] and span[0]<=docs[i]["doc_dt"]<=span[1]<=at for i in relevant),"explicit range")
        if m.get("variant"):
            orig=m["original_query_id"];require(labels[k]==labels[orig] and m["task_id"]==c[orig]["task_id"],"variant labels")
        else:
            primaries.append(k);family=m["family"]
            if m["judgment"]=="exact":
                if family in ("subject","future_subject"):
                    text=query["query"];term=text.split("«",1)[1].rsplit("»",1)[0];matches=titles[box,norm(term)]
                elif family=="sender":matches=domains[box,m["rule_id"][7:]]
                elif family in ("identifier","future_identifier","other_mailbox","order_total","tracking_number"):
                    token=re.search(r"SYN-[0-9A-F]{6,}",query["query"],re.I)[0].upper();matches=entities[box,token]
                else:raise ValueError("Unknown exact family "+family)
                expected={i for i in matches if (not docs[i]["doc_dt"] or docs[i]["doc_dt"]<=at) and (not span or (docs[i]["doc_dt"] and span[0]<=docs[i]["doc_dt"]<=span[1]))}
                require(expected==direct,"label disagreement "+k);stats["exhaustive_rule"]+=1
            if m["judgment"]=="assistant_reviewed":
                expected={i for i,d in docs.items() if d["synthetic_user"]==box and d["doc_dt"] and span[0]<=d["doc_dt"]<=span[1]}
                require(expected==set(labels[k]),"incomplete day review "+k);stats["reviewed"]+=1
            answer=m.get("answer")
            if answer:
                i=answer["doc_id"];require(i in direct and answer["evidence"] in docs[i]["doc"],"missing source span")
                stats["rag_"+answer["answerability"]]+=1
            if family=="other_mailbox":
                require(any(bb!=box and t==token and any(not docs[i]["doc_dt"] or docs[i]["doc_dt"]<=at for i in ii) for (bb,t),ii in entities.items()),"missing other-mailbox counterpart")
            require(not(direct & prior),"previously labelled target "+k)
        textset.add(norm(query["query"]));judged.update(relevant);groups.add(m["validation_group"]);mailboxes.add(box)
    panel_path=BASE/"panels"/(folder.name+".jsonl")
    if panel_path.exists():
        p=read(panel_path);require(len({v["query_id"] for v in p})==250 and all(v["query_id"] in primaries for v in p),"panel")
        require(Counter(v["task_type"] for v in p)=={"known_item":100,"collection":100,"empty":50},"panel balance")
    print(folder.name,"primary",len(primaries),"variants",len(variants),dict(stats))
    return textset,judged,groups,mailboxes,set(q)
def main():
    inputs=BASE/"method/prior_targets.json";known=json.loads(inputs.read_text()) if inputs.exists() else {}
    public=BASE/"github/data/qrels_dev.jsonl";publicgold={l["doc_id"] for l in read(public)} if public.exists() else set()
    for kind in ("personal","vk"):
        results={}
        for split in ("dev","holdout"):
            p=BASE/"suite"/f"{kind}_{split}"
            if not p.exists():continue
            if not results:packed=index(p/"documents.jsonl")
            results[split]=check(p,packed,set(known.get(kind,[]))|(publicgold if kind=="vk" else set()))
        if len(results)==2:
            for name,a,b in zip(("text","positive document","group","mailbox","query ID"),results["dev"],results["holdout"]):
                require(not a&b,"Dev/holdout overlap: "+kind+" "+name)
    print("All checks passed.")
if __name__=="__main__":main()
