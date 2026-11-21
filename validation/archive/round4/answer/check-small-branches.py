import json,re,sys
from pathlib import Path
from unittest.mock import patch
from collections import Counter
root=Path(__file__).resolve().parents[3];wt=root/'.cache/round4-answer';sys.path.insert(0,str(wt));sys.path.insert(0,str(wt/'tools'));from mailsearch import generation as g
from check_rag_facts import get
D={r['doc_id']:r for r in (json.loads(l) for l in open(root/'data/documents.jsonl'))}
for n in ('rag_facts','validation_pilot/rag_facts','rag_unseen','rag_confirmation'):
 small=json.load(open(root/'tests/fixtures'/(n+'.json')));past={r['id']:r for r in json.load(open(root/'.cache/round4/answer/combo-9a720b79d78a'/(n.replace('/','_')+'.json')))['cases']};tot=Counter();
 for c in small['cases']:
  q=c['query'];docs=[D[r['id']] for r in get('http://127.0.0.1:18817','/api/search',{'mailbox':c['mailbox'],'date':c['date'],'query':q,'limit':5})['items']];anchors=set(g._ID.findall(g._normal(q)));sel=[d for d in docs if (not anchors or anchors.intersection(g._ID.findall(g._normal(d['doc'])))) and g._is_related(q,d['doc'])];detail=g._anchored_detail(q,sel) if not g._requested_field(q) and not g._WHOLE_MAILBOX_COUNT.search(q) else None
  prev=past[c['id']]; prev_ok=prev['all_expected_facts_cited'] if c.get('facts', []) else prev['mode']=='no_evidence';
  if detail:
   ok=all(re.search(f,detail['answer'],re.I) for f in c.get('facts', [])) and bool(c.get('facts', [])) and set(detail['sources'])<=set(c['sources']) if c.get('facts', []) else detail['mode']=='no_evidence';tot['overridden']+=1
   print(n,'changed path',c['id'],'question type',len(c.get('facts', [])),'old',prev_ok,'new',ok,flush=True)
  else:ok=prev_ok
  tot['after_correct']+=bool(ok);tot['before_correct']+=bool(prev_ok)
 print(n,tot,flush=True)
