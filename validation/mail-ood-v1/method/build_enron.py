"""Build a held-out email search collection from complete mailboxes in the CMU Enron archive."""
from __future__ import annotations
from collections import Counter,defaultdict
from datetime import timedelta
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime,parseaddr
from pathlib import Path
import hashlib,json,re,tarfile,tempfile
import build_suite as common

ROOT=common.ROOT
MAILBOXES={"allen-p","bass-e","beck-s","delainey-d","farmer-d","kean-s","lay-k","skilling-j","taylor-m","thomas-p"}
RULES={
 "reschedule":(r"(?:reschedul|postpon).{0,35}(?:meeting|call|conference)|(?:meeting|call|conference).{0,35}(?:reschedul|postpon)","messages about a meeting or call being moved","переписку о переносе встречи или звонка"),
 "cancel":(r"cancel(?:led|ed|lation).{0,30}(?:meeting|call)|(?:meeting|call).{0,30}cancel","notices about a meeting or call being cancelled","уведомления об отмене встречи или звонка"),
 "agenda":(r"\bagenda\b","mail about an agenda","письма с повесткой встречи"),
 "minutes":(r"meeting minutes|minutes of (?:the )?meeting","messages about minutes from a meeting","протоколы прошедших совещаний"),
 "conference":(r"conference call|dial[\s-]?in","messages about a conference call or dial-in details","письма об участии в телефонной конференции"),
 "earnings":(r"\bearnings\b","messages discussing earnings","письма о финансовых результатах и прибыли"),
 "invoice":(r"\binvoic(?:e|es|ing)\b","messages about invoices","письма о выставленных счетах"),
 "expense":(r"expense reports?|expense statements?","messages about expense reports","письма об отчётах по расходам"),
 "weather":(r"weather (?:forecast|update|report)|weather outlook","weather updates or forecasts","письма с прогнозом погоды"),
 "maintenance":(r"\bmaintenance\b","messages about maintenance work","письма о техническом обслуживании"),
 "outages":(r"\boutages?\b","messages about outages","сообщения о перебоях или отключениях"),
 "nominations":(r"\bnominations?\b","messages about nominations","письма об упомянутых в теме номинациях или заявках на поставку"),
 "storage":(r"\bstorage\b","mail discussing storage","письма про хранение и хранилища"),
 "pipeline":(r"\bpipelines?\b","mail discussing pipelines","письма о трубопроводах"),
 "travel":(r"itinerar(?:y|ies)|travel arrangements|flight arrangements","messages about travel itineraries and travel arrangements","письма о маршруте и организации поездки"),
 "budget":(r"\bbudget\b","mail about a budget","письма об обсуждении бюджета"),
 "contracts":(r"contract (?:amendment|renewal|extension)|amendment to","mail about amendments or extension of an agreement","письма об изменении или продлении договора"),
}
def header(value):return " ".join(str(value or "").split())
def get_text(msg):
    try:
        part=msg.get_body(preferencelist=("plain","html")) if msg.is_multipart() else msg
        if part is None:return ""
        val=part.get_content()
        return val if isinstance(val,str) else ""
    except (LookupError,UnicodeError,ValueError,KeyError):
        blob=msg.get_payload(decode=True);return blob.decode("utf-8",errors="replace") if isinstance(blob,bytes) else ""
def corpus(archive,out):
    counts=Counter();parse_stats=Counter();seen=defaultdict(set);ids=defaultdict(set);parser=BytesParser(policy=policy.default)
    target=out/"documents.jsonl";source_hash=common.sha(archive);metadata=out/"archive.json"
    if target.is_file() and metadata.is_file():
        saved=json.loads(metadata.read_text())
        if saved['source_sha256']==source_hash and saved['documents_sha256']==common.sha(target):return
        raise ValueError("An existing Enron snapshot has different input or documents")
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",prefix=".enron-",dir=out,delete=False) as f:
        pending=Path(f.name)
        with tarfile.open(archive,'r|gz',bufsize=2**20) as tar:
            for obj in tar:
                tar.members.clear()
                if not obj.isfile():continue
                parts=obj.name.strip("./").split("/")
                if len(parts)<3 or parts[0]!="maildir":continue
                b=parts[1];parse_stats["archive_message_files"]+=1
                if b not in MAILBOXES:continue
                parse_stats["selected_message_files"]+=1
                with tar.extractfile(obj) as fp:msg=parser.parsebytes(fp.read())
                subj=header(msg.get("Subject"));sender=header(msg.get("From"));to=header(msg.get("To"));date_str=header(msg.get("Date"))
                try:dt=parsedate_to_datetime(date_str).date().isoformat()
                except (TypeError,ValueError,OverflowError):dt=None
                body=get_text(msg).strip();message_id=header(msg.get("Message-ID"))
                sig=common.digest("enron-message",sender,to,subj,date_str," ".join(body.split()))
                if sig in seen[b] or (message_id and message_id in ids[b]):
                    parse_stats["within_mailbox_duplicates"]+=1;continue
                seen[b].add(sig)
                if message_id:ids[b].add(message_id)
                box="enron-"+b;did="enron/"+b+"/"+sig[:24]
                f.write(json.dumps({"doc_id":did,"synthetic_user":box,"doc_dt":dt,
                                    "doc":f"От: {sender}\nТема: {subj}\nТекст письма: {body}"},ensure_ascii=False,separators=(",",":"))+"\n")
                counts[box]+=1;parse_stats["empty_body"]+=not bool(body);parse_stats["unknown_dates"]+=dt is None
    if len(counts)!=len(MAILBOXES):raise ValueError(("Missing mailboxes",set(MAILBOXES)-{b[6:] for b in counts}))
    pending.replace(target)
    common.dump(metadata,{"source":"CMU enron_mail_20150507.tar.gz","source_sha256":source_hash,"documents_sha256":common.sha(target),"selected_mailboxes":dict(counts),**parse_stats})
def build():
    out=ROOT/"external/benchmarks/enron-mail";out.mkdir(exist_ok=True,parents=True)
    archive=ROOT/"github/datasets/local/enron/enron_mail_20150507.tar.gz";corpus(archive,out)
    boxes=defaultdict(list)
    with (out/"documents.jsonl").open() as f:
        for line in f:
            r=json.loads(line);d=common.extract(r);boxes[d.box].append(d)
    for b in boxes:common.NATIVE[b]=b
    suite=common.Suite("enron",boxes,set());suite.hold=set(boxes)
    for b,rows in sorted(boxes.items()):
        bysub=defaultdict(list);byaddr=defaultdict(list)
        for d in rows:
            if common.safe_title(d.subject):bysub[common.normalize(d.subject)].append(d)
            sender=parseaddr(d.text.split("\n",1)[0][3:])[1].lower()
            if re.fullmatch(r"[-\w.+]+@[\w.-]+\.[a-z]{2,}",sender) and d.day:byaddr[sender].append(d)
        # Semantic categories use hand-written subject rules and are marked silver.
        for fam,(regex,eng,ru) in RULES.items():
            hits=[d for d in rows if re.search(regex,common.normalize(d.subject),re.I)]
            if not hits:continue
            periods=common.periods(hits,"month")
            for lo,hi in common.doc_by_hash(periods,(b,fam))[:7]:
                gold=periods[lo,hi]
                if len(gold)>35:continue
                q=f"Find {eng} received between {lo} and {hi}, inclusive."
                suite.add(b,"semantic","enron-"+fam,q,f"Покажи {ru}, полученные с {common.pretty(lo)} по {common.pretty(hi)}.",hi+common.DAY,gold,lo=lo,hi=hi,level="semantic_silver")
            if len(hits)<=35 and any(d.day for d in hits):
                at=max(d.day for d in hits if d.day)+7*common.DAY
                suite.add(b,"semantic_no_date","enron-"+fam,f"Find {eng}.",f"Найди {ru}.",at,hits,level="semantic_silver")
        # Known sender and exact subject tasks are exhaustive within a whole selected mailbox.
        n=0
        for addr in common.doc_by_hash(byaddr,b):
            matches=byaddr[addr]
            if len(matches)<3:continue
            byweek=common.periods(matches,"week")
            for lo,hi in common.doc_by_hash(byweek,addr):
                gold=byweek[lo,hi]
                if not 2<=len(gold)<=20:continue
                n+=suite.add(b,"sender","sender-"+addr,f"Show email from {addr} received during the week of {lo}.",
                             f"Найди письма от {addr} за период с {common.pretty(lo)} по {common.pretty(hi)}.",hi+common.DAY,gold,lo=lo,hi=hi,level="exact")
                break
            if n>=85:break
        n=empty=0;first_day=min(d.day for d in rows if d.day)
        for subject in common.doc_by_hash(bysub,b):
            gold=bysub[subject];days=[d.day for d in gold if d.day]
            if not days or len(gold)>12:continue
            raw=gold[0].subject;q=f'Find mail with the exact subject "{raw}".'
            if n<110:
                n+=suite.add(b,"subject","subject-"+common.digest(subject)[:12],q,f"Найди письмо с полной темой «{raw}».",max(days)+3*common.DAY,gold,level="exact")
            at=min(days)-common.DAY
            if empty<50 and at>first_day and all(d.day is not None and d.day>at for d in gold):
                empty+=suite.add(b,"future_subject","subject-before-"+common.digest(subject)[:12],f'Is there an email whose entire subject is "{raw}"?',None,at,[],level="exact",proof="Every message with this full subject is later than the as-of date.")
            if n>=110 and empty>=50:break
    counts=common.bundle(suite,"holdout",out/"documents.jsonl",output_dir=out)
    print(dict((k,counts[k]) for k in ("documents","primary_tasks","queries","empty","variants","families")),flush=True)
if __name__=="__main__":build()
