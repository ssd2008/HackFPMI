from pathlib import Path
from collections import Counter
import json,os,re,hashlib,sys
root=Path(__file__).resolve().parents[3];wt=root/'.cache/round4-answer';sys.path.insert(0,str(wt));sys.path.insert(0,str(wt/'tools'));from mailsearch import generation as g
from check_mail_ood_rag import load_inputs,grade
pool=Path('/Users/kirill/Downloads/HackFPMI-mail-OOD');cache=root/'.cache/ood-mail/combo-9a720b79d78a';out={}
for split in ['dev','holdout']:
 for src in ['vk','personal']:
  data=load_inputs(pool,src,split);d=data['docs'];ranks=json.load(open(cache/split/src/'top5.json'));changes=Counter()
  for cid,c in data['cases'].items():
   q=c['query'];w=data['metadata'][cid];sel=[d[i] for i in ranks[cid]];anchors=set(g._ID.findall(g._normal(q)));sel=[doc for doc in sel if (not anchors or anchors.intersection(g._ID.findall(g._normal(doc['doc'])))) and g._is_related(q,doc['doc'])];detail=g._anchored_detail(q,sel) if not g._requested_field(q) and not g._WHOLE_MAILBOX_COUNT.search(q) else None
   prev=cache/split/src/'after'/(hashlib.sha256(cid.encode()).hexdigest()+'.json');old=grade(json.loads(prev.read_bytes()),c,w,d)['class'] if prev.is_file() else 'not_yet';
   if detail:
    record={'mode':detail['mode'],'answer':detail['answer'],'opened_source_ids':detail['sources'],'answer_http_ms':0,'citations_to_expected':len(set(c['sources'])&set(detail['sources']))};after=grade(record,c,w,d)['class'];changes['intercepted']+=1;changes[old+' -> '+after]+=1
   else:after=old
   if w['family']=='reviewed_semantic':print(split,src,'item',list(data['cases']).index(cid)+1,old,' => ',after, 'field', bool(detail),flush=True)
  out[(split,src)]=changes
print(out)
