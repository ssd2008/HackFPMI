"""Paired VK checks for undated mail and for conditional versus completed events."""
from collections import Counter,defaultdict
from datetime import date,timedelta
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
def rows(path):
    with path.open() as f:return [json.loads(l) for l in f if l.strip()]
def order(s):return hashlib.sha256(("event-pairs-v1/"+s).encode()).digest()
def norm(s):return " ".join(s.casefold().replace("ё","е").split())
def main():
    raw=rows(ROOT/"github/data/documents.jsonl");bybox=defaultdict(list);seen=set()
    for d in raw:bybox[d["synthetic_user"]].append(d)
    for a in list((ROOT/"suite").glob("vk_*/qrels_dev.jsonl"))+[ROOT/"github/data/qrels_dev.jsonl"]:
        seen.update(x["doc_id"] for x in rows(a) if x["relevance"]>0)
    out=ROOT/"diagnostics";out.mkdir(exist_ok=True)
    pos=[];empty=[];links=[];cases=[]
    def pair(b,at,target,k,q1,q2,reason):
        n="E-"+order(b+k+target['doc_id']).hex()[:16]
        for kind,dest,query in (("positive",pos,q1),("verified_empty",empty,q2)):
            qid=n+("-p" if kind=="positive" else "-e")
            dest.append({"query_id":qid,"synthetic_user":b,"query":query,"query_date":at})
            cases.append({"query_id":qid,"pair":n,"family":k,"state":kind,"source_document":target["doc_id"],"reason":reason})
            if kind=="positive":links.append({"query_id":qid,"doc_id":target["doc_id"],"relevance":2})
    for b,docs in bybox.items():
        literals={d['doc_id']:norm(d['doc']) for d in docs};ordered=sorted(docs,key=lambda d:order(d['doc_id']))
        ids=defaultdict(list)
        for d in docs:
            for x in set(re.findall(r'(?<![\w-])SYN-[a-f0-9]{6,}(?![\w-])',d['doc'],re.I)):ids[x.upper()].append(d)
        latest=max(x['doc_dt'] for x in docs if x['doc_dt'])
        n=0
        for d in ordered:
            if n>=10:break
            if d['doc_id'] in seen or not d['doc_dt']:continue
            text=literals[d['doc_id']]
            if not re.search(r'если.{0,34}не обработать.{0,30}(?:отменят|отменен)',text):continue
            orig=re.search(r'\bзаказ\s*(?:номер |№\s*)?(SYN-[a-f0-9]{6,})(?![\w-])',d['doc'],re.I)
            if not orig or len(ids[orig[1].upper()])!=1 or re.search(r'(?:покупатель отменил|уже отменен|заказ был отменен)',text):continue
            token=orig[1]
            pair(b,latest,d,"conditional_cancellation",
                 f"Где предупреждают, что заказ {token} могут отменить, если его не обработать?",
                 f"Найди подтверждение, что заказ {token} уже отменили.",
                 "В ящике одно письмо с этим номером, и в нём отмена обусловлена будущим бездействием.")
            n+=1
        n=0
        for d in ordered:
            if n>=8:break
            if d['doc_id'] in seen or d['doc_dt'] is not None or "покупатель отменил заказ" not in literals[d['doc_id']]:continue
            body=re.split('(?:текст письма:|покупатель передумал)',literals[d['doc_id']],maxsplit=1)[-1]
            # The quoted passage must occur verbatim in this email alone, including all other undated messages.
            chunks=re.findall(r'(?=(\b[а-яa-z0-9-]{2,}(?: [а-яa-z0-9-]{2,}){4,6}\b))',body)
            phrase=next((s for s in sorted(set(chunks),key=order) if len(s)<=90 and s in literals[d["doc_id"]]
                         and sum(s in text for text in literals.values())==1),None)
            if not phrase:continue
            pair(b,latest,d,"unknown_receipt_date",
                 f"Найди письмо с точной фразой «{phrase}».",
                 f"Какие письма с точной фразой «{phrase}» пришли вчера?",
                 "Единственное совпадение в ящике имеет doc_dt=null. Для конкретного дня дата не подтверждена.")
            n+=1
    for name,items in (("queries.jsonl",pos+empty),("qrels.jsonl",links),("pairs.jsonl",cases)):
        with (out/name).open("w") as f:
            for v in items:f.write(json.dumps(v,ensure_ascii=False)+"\n")
    print(len(pos),"pairs",dict(Counter(r["family"] for r in cases if r["state"]=="positive")))
if __name__=="__main__":main()
