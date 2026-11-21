"""Build the local mail retrieval suites using only the frozen documents and saved rules."""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from email.utils import parseaddr
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
SEED = "hackfpmi-ood-v1"
DAY = timedelta(days=1)
MONTHS = "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split()
NATIVE = {"gmail": "Кирилл", "yandex": "Кирилл", "box-37e4ff4a5a57": "Сергей"}
IDENT = re.compile(r"(?<![\w-])SYN-[0-9a-f]{6,}(?![\w-])", re.I)
RE_ORDER = re.compile(r"(?<!\w)заказ(?:а|у)?\s*(?:№\s*|номер\s*)?(SYN-[0-9a-f]{6,})(?![\w-])", re.I)
MONEY = re.compile(r"(?:общ(?:ая|ую)\s+сумм[ау]\s*(?:заказа\s*)?|на\s+сумму\s*)[—–:\-\s]*(\d+(?:[ \u00a0]\d{3})*[.,]\d{2})\s*(RUB|руб(?:лей|ля)?\.?|₽)(?!\w)",re.I)
LEFT_HEADER = re.compile(r"^\s*(?:тема|subject)\s*:[ \t]*(.*)$",re.I|re.M)
FROM_HEADER = re.compile(r"^\s*(?:от|from|отправитель)\s*:[ \t]*(.*)$",re.I|re.M)
UNSUITABLE = re.compile(r"парол|password|verif|confirm|reset|подтверд|подтвержд|аккаунт|security|код доступа|код.{0,15}(?:вход|регистрац)|ваш код|одноразов|access token|логин|token|@\w|https?://|www\.|[A-F\d]{9,}|(?:\d[\s-]*){6,}|судимост|поликлиник|паспорт|болезн|здоровь|медицин|анализ крови|енис|справка|зарплат|взыскан|резюме|задолж|секрет|оплат.+карт|личн.+данн|кредит|транзакц|парол|withheld|скрыто|банкрот",re.I)
def read(path):
    with path.open(encoding="utf-8") as inp:
        return [json.loads(line) for line in inp if line.strip()]
def put(path, rows):
    with path.open("w",encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
def dump(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def digest(*items):
    return hashlib.sha256("\0".join((SEED,*map(str,items))).encode()).hexdigest()
def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()
def normalize(s):
    return " ".join(unicodedata.normalize("NFKC",s).casefold().replace("ё","е").split())
def pretty(dt):
    return f"{dt.day} {MONTHS[dt.month-1]} {dt.year} года"
def doc_by_hash(items, key):
    return sorted(items,key=lambda r:digest(key,r))
@dataclass
class Doc:
    id:str
    box:str
    day:date|None
    text:str
    subject:str
    domain:str
    def __hash__(self):
        return hash(self.id)
def extract(row):
    text=row["doc"]
    head=re.split(r"(?:^|\n)[ \t]*текст письма\s*:",text,1,flags=re.I)[0][:2000]
    m=LEFT_HEADER.search(head)
    subject=m[1].strip() if m else ""
    m=FROM_HEADER.search(head)
    addr=parseaddr(m[1])[1].casefold() if m else ""
    domain=addr.rsplit("@",1)[-1].rstrip("> ") if "@" in addr else ""
    return Doc(row["doc_id"],row["synthetic_user"],date.fromisoformat(row["doc_dt"]) if row["doc_dt"] else None,text,subject,domain)
def attach(source,target):
    if target.exists():
        if os.path.samefile(source,target) or sha(source)==sha(target):return
        raise ValueError(f"Existing documents differ at {target}")
    os.link(source,target)
def visible(d,at):
    return d.day is None or d.day<=at
def span_words(lo,hi):
    return pretty(lo) if lo==hi else f"с {pretty(lo)} по {pretty(hi)}"
def safe_title(subject):
    s=subject.strip()
    if not 12<=len(s)<=90 or len(s.split())<3 or UNSUITABLE.search(s) or re.search(r'(?:[<>{}=]|(?:[А-ЯЁ]\.\s*){2}|[\r\n])',s):return False
    return bool(re.search(r"[a-zа-я]{3}",s,re.I))
class Suite:
    def __init__(self, kind, bybox, public_gold):
        self.kind=kind
        self.bybox=bybox
        self.byid={d.id:d for rows in bybox.values() for d in rows}
        self.public_gold=public_gold
        self.cases=defaultdict(list)
        self.seen=set()
        self.used_sets=set()
        self.primary_text=set()
        self.ids=set()
        boxes=sorted(bybox,key=lambda b:digest("vk-box",b))
        if kind=="vk":
            # Reserve whole mailboxes, including ones with no organizer queries.
            withgold={self.byid[i].box for i in public_gold if i in self.byid}
            self.hold=set([b for b in boxes if b in withgold][:3]+[b for b in boxes if b not in withgold][:2])
        else:
            self.hold={"gmail","yandex"}
    def split(self,b):
        return "holdout" if b in self.hold else "dev"
    def add(self,b,family,rule,query,alt,at,positives,*,lo=None,hi=None,proof=None,task_type=None,negatives=(),rag=None,level="rule",priority_key=None,typo=None):
        pos=sorted({d.id for d in positives})
        neg=sorted({d.id for d in negatives}-set(pos))
        if any(i in self.public_gold for i in pos):return False
        if not pos and not proof:raise ValueError("Empty without proof")
        if any(self.byid[i].box!=b or not visible(self.byid[i],at) for i in pos+neg):raise ValueError("Out-of-scope label")
        if lo and any(self.byid[i].day is None or not lo<=self.byid[i].day<=hi for i in pos):raise ValueError("Out-of-window label")
        # Collection, sender and known-item variants returning the same targets do not inflate the primary count.
        key=(b,tuple(pos)) if pos else (b,rule,lo or at,hi or at)
        if key in self.used_sets:return False
        if (b,normalize(query),at) in self.seen:return False
        self.used_sets.add(key)
        self.seen.add((b,normalize(query),at))
        task="T-"+digest(self.kind,b,family,rule,at,query)[:18]
        if task in self.ids:raise ValueError("Duplicate task")
        self.ids.add(task)
        query_id="Q-"+task[2:]
        group=b if self.kind=="vk" else NATIVE[b]
        meta={"query_id":query_id,"task_id":task,"source":self.kind,"validation_group":group,"family":family,"rule_id":rule,"status":"positive" if pos else "verified_empty","task_type":task_type or ("single" if len(pos)==1 else "collection" if pos else "empty"),"judgment":level,"relevant_count":len(pos)}
        anchor=rule.replace("subject-before-","subject-").replace("id-before-","id-").replace("outside-SYN","id-SYN")
        meta["intent_group"]=b+":"+anchor
        meta["family_id"]=meta["intent_group"]
        meta["split"]="blind" if self.split(b)=="holdout" else "dev"
        meta["state"]=meta["status"]
        if meta["task_type"]=="single":
            meta["task_type"]="known_item"
            meta["target_ids"]=pos
        if lo:meta["receipt_window"]=[str(lo),str(hi)]
        if proof:meta["proof"]=proof
        if rag:meta["answer"]=rag
        queryrow={"query_id":query_id,"synthetic_user":b,"query":query,"query_date":str(at)}
        alternatives=[]
        for style,v in [("paraphrase",alt),("typo",typo)]:
            if v and normalize(v)!=normalize(query):
                vid=query_id+("-P" if style=="paraphrase" else "-E")
                alternatives.append((dict(queryrow,query_id=vid,query=v),style))
        self.cases[self.split(b)].append((queryrow,meta,pos,neg,alternatives))
        return True
def topic_rules():
    out=[]
    for ln in (ROOT/"method/topics.tsv").read_text().splitlines():
        if not ln or ln.startswith("#"):continue
        tag,kind,domain,field,rest=ln.split("|",4)
        patt,word,alt=rest.rsplit("|",2)
        out.append((tag,kind,domain,field,re.compile(patt,re.I),word,alt))
    return out
def typo_variant(s):
    # A single plausible letter transposition outside names, literal IDs and dates.
    for old,new in (("письма","писмьа"),("Найди","Нади"),("Покажи","Покжаи"),("уведомления","уведмоления"),("сообщения","сообшения")):
        if old in s:return s.replace(old,new,1)
    return None
def periods(rows,mode):
    grouped=defaultdict(list)
    for d in rows:
        if not d.day:continue
        dt=d.day
        if mode=="day":lo=hi=dt
        elif mode=="week":
            lo=dt-timedelta(days=dt.weekday());hi=lo+6*DAY
        else:
            lo=dt.replace(day=1);hi=(lo.replace(day=28)+4*DAY).replace(day=1)-DAY
        grouped[(lo,hi)].append(d)
    return grouped
def groups(suite,b,rows,rule,word,alt,matching,family,limit,level):
    for mode in ("week","month","day"):
        put_count=0
        windows=periods(matching,mode)
        for lo,hi in doc_by_hash(windows,(b,rule,mode)):
            gold=windows[(lo,hi)]
            if len(gold)>35 or (mode!="day" and len(gold)<2):continue
            at=hi+DAY
            rel=mode=="day" and int(digest(rule,b,lo)[:8],16)%2==0
            phrase="вчера" if rel else span_words(lo,hi)
            if mode=="week" and int(digest(rule,lo)[:8],16)%3==0:phrase="за прошлую календарную неделю"
            if mode=="month" and int(digest(rule,lo,hi)[:8],16)%3==0:phrase="за прошлый календарный месяц"
            q=f"Найди {word}, полученные {phrase}." if not phrase.startswith("за ") else f"Найди {word} {phrase}."
            aq=f"Покажи {alt} за {span_words(lo,hi)}." if mode=="day" else f"Покажи {alt} за период с {pretty(lo)} по {pretty(hi)}."
            if mode=="day" and rel:aq=f"Покажи {alt} за вчера."
            eligible=[d for d in rows if d.day and lo<=d.day<=hi]
            # Include up to three in-window alternatives for spot checks.
            eg={x.id for x in gold}
            negatives=sorted((d for d in eligible if d.id not in eg),key=lambda d:digest(rule,d.id))[:3] if level=="exact" else []
            if suite.add(b,family,rule,q,aq,at,gold,lo=lo,hi=hi,negatives=negatives,level=level,typo=typo_variant(q) if put_count%5==0 else None):
                put_count+=1
                if put_count>=limit:break
def semantic(suite,b,rows):
    normalized={d.id:(normalize(d.subject),normalize(d.text)) for d in rows}
    for tag,kind,domain,field,regex,word,alt in topic_rules():
        if kind!=suite.kind:continue
        matching=[d for d in rows if (domain=="*" or d.domain==domain) and regex.search(normalized[d.id][field=="body"])]
        if not matching:continue
        groups(suite,b,rows,tag,word,alt,matching,"semantic",16,"semantic_silver")
        # The snapshot is the only time limit: the user did not supply a receipt date.
        days=sorted({d.day for d in matching if d.day})
        for idx in (2,7,18,len(days)-1):
            if idx<0 or idx>=len(days):continue
            at=days[idx]+2*DAY
            hits=[d for d in matching if visible(d,at)]
            if 1<=len(hits)<=50:
                for q in (f"Мне нужны {word}.", f"Найди {word}."):
                    if suite.add(b,"semantic_no_date",tag,q,f"Покажи {alt}.",at,hits,level="semantic_silver"):break
def senders(suite,b,rows):
    bydom=defaultdict(list)
    for d in rows:
        if re.fullmatch(r"[a-z0-9][a-z0-9.-]+\.[a-z]{2,}",d.domain):bydom[d.domain].append(d)
    for domain,matching in sorted(bydom.items()):
        if len(matching)<5 or domain in {"gmail.com","yandex.ru","mail.ru","bk.ru","outlook.com","example.invalid"}:continue
        word=f"письма с адресов на @{domain}"
        alt=f"сообщения, у которых домен отправителя — {domain}"
        groups(suite,b,rows,"sender-"+domain,word,alt,matching,"sender",4,"exact")
def titles(suite,b,rows):
    bytitle=defaultdict(list)
    for d in rows:
        if safe_title(d.subject):bytitle[normalize(d.subject)].append(d)
    count=empty=0
    for term in doc_by_hash(bytitle,(b,"subject")):
        matches=bytitle[term]
        if any(d.id in suite.public_gold for d in matches):continue
        dates=[d.day for d in matches if d.day]
        if not dates:continue
        first,last=min(dates),max(dates)
        subject=matches[0].subject
        if count<460:
            at=last+timedelta(days=3+int(digest(term)[:2],16)%40)
            q=f"Найди письмо с темой «{subject}»."
            alt=f"В каком письме заголовок целиком совпадает с «{subject}»?"
            count+=suite.add(b,"subject","subject-"+digest(term)[:12],q,alt,at,matches,level="exact")
        pre=first-timedelta(days=1+int(digest(term)[:2],16)%8)
        if empty<150 and len(matches)<=4 and sum(d.day is not None and d.day<=pre for d in rows)>=20 and not any(visible(d,pre) for d in matches):
            q=f"Найди письмо, у которого тема целиком «{subject}»."
            empty+=suite.add(b,"future_subject","subject-before-"+digest(term)[:12],q,None,pre,[],proof="Все сообщения с этой полной темой в ящике датированы позже query_date.",level="exact")
        if count>=460 and empty>=150:break
def refs_and_answers(suite,b,rows):
    byref=defaultdict(list)
    for d in rows:
        for token in {m[0].upper() for m in IDENT.finditer(d.text)}:
            byref[token].append(d)
    count=empty=0
    for token in doc_by_hash(byref,b):
        matching=byref[token]
        if len(matching)>10 or any(d.id in suite.public_gold for d in matching):continue
        dates=[d.day for d in matching if d.day]
        if not dates:continue
        first,last=min(dates),max(dates)
        at=last+timedelta(days=5+int(digest(token)[:2],16)%45)
        if count<250:
            rag=None
            if len(matching)==1:
                d=matching[0];orders={m[1].upper() for m in RE_ORDER.finditer(d.text)}
                amounts=list(MONEY.finditer(d.text))
                if orders=={token} and len(amounts)==1:
                    amount=amounts[0];value=re.sub(r"\s","",amount[1]).replace(",",".")
                    rag={"answerability":"answerable","fact":"order_total","value":value,"currency":"RUB","doc_id":d.id,"evidence":amount[0],"entity":token}
            if rag and int(digest(token,"missing")[:8],16)%5==0 and not re.search(r"трек|отслеж|почтовый идентификатор|tracking|track.number|номер.{0,12}(?:отправления|накладной)",matching[0].text,re.I):
                rag={"answerability":"unanswerable","fact":"tracking_number","entity":token,"doc_id":matching[0].id,"reason":"Единственное письмо с номером заказа не называет трек-номер.","evidence":next(m[0] for m in IDENT.finditer(matching[0].text) if m[0].upper()==token)}
            if rag and rag["answerability"]=="unanswerable":
                q=f"Какой трек-номер отправления указан в письме о заказе {token}?";alt=f"Из письма про заказ {token} можно узнать почтовый идентификатор?"
            elif rag:
                q=f"На какую сумму был заказ {token}?";alt=f"Сколько рублей составил заказ {token}?"
            else:
                q=f"В каких письмах встречается идентификатор {token}?";alt=f"Покажи переписку, в которой упоминается номер {token}."
            count+=suite.add(b,rag["fact"] if rag else "identifier","id-"+token,q,alt,at,matching,rag=rag,level="exact")
        pre=first-timedelta(days=1+int(digest(token)[:2],16)%4)
        if empty<70 and any(d.day and d.day<=pre for d in rows) and not any(visible(d,pre) for d in matching):
            q=f"Есть письмо, где упоминается номер {token}?"
            empty+=suite.add(b,"future_identifier","id-before-"+token,q,None,pre,[],proof="Идентификатор проверен по всем текстам ящика. Все совпадения датированы позже query_date.",level="exact")
        if count>=250 and empty>=70:break
def cross_box(suite):
    tokens=defaultdict(dict)
    for b,rows in suite.bybox.items():
        for d in rows:
            for m in IDENT.finditer(d.text):
                if d.day:tokens[m[0].upper()].setdefault(b,d.day)
    for b,rows in suite.bybox.items():
        n=0;at=max(d.day for d in rows if d.day)
        own={m[0].upper() for d in rows for m in IDENT.finditer(d.text)}
        for t in doc_by_hash(tokens,(b,"outside")):
            if t in own or not any(v<=at for v in tokens[t].values()):continue
            q=f"Найди у меня переписку с номером {t}."
            n+=suite.add(b,"other_mailbox","outside-"+t,q,None,at,[],proof="В ящике нет этого точного идентификатора. На дату запроса он встречается в другом ящике.",level="exact")
            if n>=40:break
def run_core(suite):
    p=ROOT/"method/reviewed.jsonl"
    if not p.is_file():return
    for r in read(p):
        b=r["box"]
        if b not in suite.bybox:continue
        rows=suite.bybox[b];at=date.fromisoformat(r["at"])
        lo=date.fromisoformat(r["lo"]);hi=date.fromisoformat(r["hi"])
        eligible=[d for d in rows if d.day and lo<=d.day<=hi]
        pos=[suite.byid[i] for i in r["ids"]]
        context=set(r.get("context",[]))
        if not context.issubset({d.id for d in eligible}) or context & {d.id for d in pos}:raise ValueError("Invalid graded context")
        if (context | {d.id for d in pos}) & suite.public_gold:
            continue
        neg=[d for d in eligible if d not in pos and d.id not in context]
        ok=suite.add(b,"reviewed_semantic",r["id"],r["query"],r.get("alternate"),at,pos,lo=lo,hi=hi,negatives=neg,proof=None if pos else "Прочитаны все письма ящика за указанное окно.",level="assistant_reviewed",rag=r.get("answer"))
        if not ok:raise ValueError("Duplicated or overlapping manually reviewed case "+r["id"])
        if context:
            m=suite.cases[suite.split(b)][-1][1]
            m["context_ids"]=sorted(context)
            m["target_ids"]=sorted(d.id for d in pos)

def scope_holdout(suite):
    def anchor(m):
        key=m["rule_id"]
        if key.startswith(("id-before-","id-","outside-")):
            return "entity-"+key[key.index("SYN-"):]
        if key.startswith("subject-"):
            return "title-"+key.rsplit("-",1)[-1]
        return ""
    train_text=set()
    train_anchors=set()
    for q,m,_,__,alts in suite.cases["dev"]:
        train_text.update(normalize(x["query"]) for x in [q]+[v[0] for v in alts])
        token=anchor(m)
        if token:train_anchors.add(token)
    held=[]
    for row in suite.cases["holdout"]:
        q,m,_,__,alts=row
        texts=[q]+[v[0] for v in alts]
        if anchor(m) not in train_anchors and all(normalize(x["query"]) not in train_text for x in texts):held.append(row)
    suite.cases["holdout"]=held

def bundle(suite,split,source,output_dir=None):
    out=output_dir or ROOT/"suite"/f"{suite.kind}_{split}"
    out.mkdir(parents=True,exist_ok=True)
    attach(source,out/"documents.jsonl")
    tasks=sorted(suite.cases[split],key=lambda t:t[0]["query_id"])
    positives=[];empty=[];variants=[];links=[];empty_links=[];vlinks=[];metadata=[];review=[]
    for q,m,pos,neg,alts in tasks:
        qid=q["query_id"];(positives if pos else empty).append(q);metadata.append(m)
        labels=[{"query_id":qid,"doc_id":i,"relevance":2} for i in pos]+[{"query_id":qid,"doc_id":i,"relevance":1} for i in m.get("context_ids",[])]+[{"query_id":qid,"doc_id":i,"relevance":0} for i in neg]
        (links if pos else empty_links).extend(labels)
        for v,style in alts:
            variants.append(v);vlinks.extend(dict(x,query_id=v["query_id"]) for x in labels)
            metadata.append(dict(m,query_id=v["query_id"],variant=style,original_query_id=qid))
        if split=="dev" and m["judgment"] in {"semantic_silver","assistant_reviewed"} and int(digest(qid,"review")[:8],16)%5==0:
            review.append({"query_id":qid,"synthetic_user":q["synthetic_user"],"query_date":q["query_date"],"query":q["query"],"reviewer_grade":None})
    put(out/"queries_dev.jsonl",positives);put(out/"qrels_dev.jsonl",links)
    put(out/"queries_empty.jsonl",empty);put(out/"qrels_empty.jsonl",empty_links);put(out/"queries_variants.jsonl",variants);put(out/"qrels_variants.jsonl",vlinks);put(out/"cases.jsonl",metadata)
    if split=="dev":put(out/"review_queue.jsonl",review)
    counts={"documents":len(suite.byid),"queries":len(positives),"qrels":len(links),"primary_tasks":len(tasks),"empty":len(empty),"variants":len(variants),"reviewed_primary":sum(t[1]["judgment"]=="assistant_reviewed" for t in tasks),"semantic_silver":sum(t[1]["judgment"]=="semantic_silver" for t in tasks),"rag_answerable":sum(t[1].get("answer",{}).get("answerability")=="answerable" for t in tasks),"rag_unanswerable":sum(t[1].get("answer",{}).get("answerability")=="unanswerable" for t in tasks),"families":dict(Counter(t[1]["family"] for t in tasks)),"positive_links":sum(len(t[2]) for t in tasks),"positive_documents":len(set(x for t in tasks for x in t[2])),"mailboxes":sorted({q["synthetic_user"] for q, *_ in tasks}),"single":sum(len(t[2])==1 for t in tasks),"multiple":sum(len(t[2])>1 for t in tasks)}
    dump(out/"manifest.json",{**counts,"sha256":{p.name:sha(p) for p in sorted(out.glob("*.jsonl"))}})
    return counts
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources",nargs="+",choices=["vk","personal"],default=["vk","personal"])
    args=parser.parse_args()
    for kind in args.sources:
        source=ROOT/("github/data/documents.jsonl" if kind=="vk" else "ood/documents.jsonl")
        raw=read(source);bybox=defaultdict(list)
        for x in raw:
            d=extract(x);bybox[d.box].append(d)
        pub={r["doc_id"] for r in read(ROOT/"github/data/qrels_dev.jsonl")} if kind=="vk" else set()
        known=json.loads((ROOT/"method/prior_targets.json").read_text())
        suite=Suite(kind,bybox,pub)
        suite.public_gold=pub.union(known[kind])
        run_core(suite)
        for b,rows in sorted(bybox.items()):
            semantic(suite,b,rows)
            if kind=="personal":
                senders(suite,b,rows);titles(suite,b,rows)
            else:refs_and_answers(suite,b,rows)
        if kind=="vk":cross_box(suite)
        scope_holdout(suite)
        for split in ("dev","holdout"):
            print(kind,split,json.dumps(bundle(suite,split,source),ensure_ascii=False))
if __name__=="__main__":
    main()
