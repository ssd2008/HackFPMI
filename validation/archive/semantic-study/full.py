from panels import W,H,D,HERE,get,jls
from pathlib import Path
from datetime import date
import sys,json,time,os
from collections import defaultdict
name,side=sys.argv[1:3];sys.path.insert(0,str(HERE/'pinned'/side))
from mailsearch.dense import _index_name
from mailsearch.retrieval import SearchEngine
def main():
 path=get(name);docs=jls(path/'documents.jsonl'); idx=_index_name(docs)
 options=[H/'.cache/mailsearch',H/'.cache/ood-transfer-large/vectors'/name,H/'.cache/ood-transfer-small/vectors'/name,HERE/'vectors'/name]
 cache=next((v for v in options if (v/idx).is_file()),None)
 if cache is None:raise FileNotFoundError(idx)
 engine=SearchEngine(docs,cache);allowed={d['doc_id']:(d['synthetic_user'],d['doc_dt']) for d in docs}
 orig={q['query_id']:q for q in jls(path/'queries_dev.jsonl')};kw=['dev']
 if name in ('vk_dev','personal_dev','enron-mail'):kw+=['empty','variants']
 qs={q['query_id']:q for k in kw for q in jls(path/f'queries_{k}.jsonl')}
 sets=json.loads((HERE/'panel.json').read_bytes())[name];primary=sets['full']
 if name in ('vk_dev','personal_dev'):extras=[q for q in qs if q not in primary]
 elif name=='enron-mail':
  own=set(primary);cases=jls(path/'cases.jsonl')
  extras=[r['query_id'] for r in cases if r.get('original_query_id') in own or r['state']=='verified_empty']
 else: extras=[]
 order=sorted(set(primary+extras),key=lambda x:__import__('hashlib').sha256(('semantic-run|'+x).encode()).digest())
 ranks={};duration=[]
 for cnt,qid in enumerate(order,1):
  q=qs[qid]; start=time.perf_counter_ns()
  rank=[d for d,_ in engine.search(q['synthetic_user'],q['query'],date.fromisoformat(q['query_date']),10)]
  duration.append((time.perf_counter_ns()-start)/1e6)
  assert len(set(rank))==len(rank)<=10
  for did in rank:
   box,day=allowed[did];assert q['synthetic_user']==box and(not day or day<=q['query_date'])
  ranks[qid]=rank
 print('finished',name,side,len(order),'scope checked',sum(map(len,ranks.values())),flush=True)
 (HERE/f'{side}-full-{name}.json').write_text(json.dumps({'ranks':ranks,'timing_ms':duration}))
if __name__=='__main__':main()
