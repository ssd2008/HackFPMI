import hashlib, json, random, re, statistics, sys
from collections import Counter,defaultdict
from datetime import date
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen
from spelling import CorpusSpelling
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from benchmark import metrics,percentiles,fetch
from mailsearch.retrieval import _stem
from mailsearch.dates import parse_date_scope

BASE='http://127.0.0.1:18765'
FROZEN='de1e76c49e28408cf7831ef8f77b5f40db8457fed58834b2b5f9ab8457652f3d'

def jsonl(path):
 return [json.loads(l) for l in path.read_text().splitlines()]
def dump(name,data):
 (HERE/name).write_text(json.dumps(data,ensure_ascii=False,indent=2))

raw=(HERE/'typos99.jsonl').read_bytes()
assert hashlib.sha256(raw).hexdigest()==FROZEN
clean=jsonl(ROOT/'data/queries_dev.jsonl')
typos=jsonl(HERE/'typos99.jsonl')
subs=json.loads((ROOT/'search_bench/paraphrases.json').read_text())
parap=[]
for q in clean:
 matches=[(o,n) for o,n in subs.items() if o in q['query']]
 assert len(matches)==1
 old,new=matches[0]
 parap.append({**q,'query':q['query'].replace(old,new)})
for a,b in zip(clean,typos):
 assert a['query_id']==b['query_id']
 assert parse_date_scope(a['query'],date.fromisoformat(a['query_date'])).start==parse_date_scope(b['query'],date.fromisoformat(b['query_date'])).start

docs=jsonl(ROOT/'data/documents.jsonl');data={r['doc_id']:r for r in docs}
t=perf_counter(); model=CorpusSpelling((d['doc'] for d in docs),_stem); build=perf_counter()-t
print('build',round(build,3),'forms',len(model.known),'frequent',len(model.frequent),'roots',len(model.roots),flush=True)
groups={'clean':clean,'paraphrases':parap,'typos':typos}
props=json.load(urlopen(BASE+'/api/mailboxes',timeout=5))
negbox=props['default_mailbox'];negdate=props['default_date']
neg=[{'query_id':str(i),'query':q,'synthetic_user':negbox,'query_date':negdate}
     for i,q in enumerate(json.loads((HERE/'negatives.json').read_text()))]
groups['negative']=neg
tasks=[(label,q) for label,qs in groups.items() for q in qs]
random.Random(80315).shuffle(tasks)
results=defaultdict(dict)
for idx,(group,q) in enumerate(tasks):
 raw=q['query']
 start=perf_counter(); value=model.correct(raw); normalize_ms=(perf_counter()-start)*1000
 d=date.fromisoformat(q['query_date'])
 old_day=parse_date_scope(raw,d);new_day=parse_date_scope(value,d)
 assert (old_day.start,old_day.end)==(new_day.start,new_day.end)
 row={'original':raw,'corrected':value,'normalize_ms':normalize_ms}
 arms=[('raw',raw),('spell',value)]
 if idx%2:arms.reverse()
 for arm,query in arms:
  ids,server,wall=fetch(BASE,{**q,'query':query},50)
  for did in ids:
   doc=data[did]; assert doc['synthetic_user']==q['synthetic_user'] and (not doc['doc_dt'] or doc['doc_dt']<=q['query_date'])
  row[arm]={'ids':ids,'server_ms':server,'wall_ms':wall}
 if raw==value: assert row['raw']['ids']==row['spell']['ids'],q['query_id']
 results[group][q['query_id']]=row
 if (idx+1)%100==0: print('paired',idx+1,flush=True)
dump('ranks.json',results)
# Only score after all treatments have been fixed and all retrieval requests completed.
gold=defaultdict(dict)
for r in jsonl(ROOT/'data/qrels_dev.jsonl'):gold[r['query_id']][r['doc_id']]=r['relevance']
report={'frozen_typos_sha256':FROZEN,'backend':BASE,'build_seconds':build,'known_forms':len(model.known),
        'frequent_surfaces':len(model.frequent),'known_stems':len(model.roots),'by_group':{}}
for group,items in results.items():
 obj={'queries':len(items),'corrected_queries':sum(r['original']!=r['corrected'] for r in items.values()),
      'changed_ranks':sum(r['raw']['ids']!=r['spell']['ids'] for r in items.values()),
      'normalize_ms':percentiles([r['normalize_ms'] for r in items.values()])}
 if group=='negative':
  obj['rewritten']=[(r['original'],r['corrected']) for r in items.values() if r['original']!=r['corrected']]
 else:
  obj['quality']={}
  for arm in ('raw','spell'):
   obj['quality'][arm]={}
   for k in (1,5,10,20,50):
    vs=defaultdict(list)
    for qid,row in items.items():
     for metric,value in metrics(row[arm]['ids'],gold[qid],k).items():vs[metric].append(value)
    obj['quality'][arm][str(k)]={m:statistics.mean(vals) for m,vals in vs.items()}
   obj['quality'][arm]['server_ms']=percentiles([row[arm]['server_ms'] for row in items.values()])
  deltas=[metrics(row['spell']['ids'],gold[i],10)['ndcg']-metrics(row['raw']['ids'],gold[i],10)['ndcg'] for i,row in items.items()]
  obj['ndcg10_paired']={'higher':sum(v>1e-8 for v in deltas),'tied':sum(abs(v)<=1e-8 for v in deltas),'lower':sum(v< -1e-8 for v in deltas)}
 if group=='typos':
  typed={q['query_id']:q for q in typos}
  obj['exactly_restored']=sum(row['corrected']==typed[qid]['clean_query'] for qid,row in items.items())
  obj['edited_but_not_exact']=[(typed[qid]['typed_word'],typed[qid]['correct_word'],row['corrected']) for qid,row in items.items()
                              if row['corrected']!=row['original'] and row['corrected']!=typed[qid]['clean_query']]
 report['by_group'][group]=obj
 print(group,json.dumps({**{k:v for k,v in obj.items() if k not in ('quality','edited_but_not_exact')},
                        'quality10':{k:v['10'] for k,v in obj.get('quality',{}).items()}},ensure_ascii=False),flush=True)
samples=[]
for _ in range(7):
 for _,q in tasks:
  t=perf_counter();model.correct(q['query']);samples.append((perf_counter()-t)*1000)
report['warm_normalize_ms']=percentiles(samples)
dump('report.json',report)
print('warm',report['warm_normalize_ms'],flush=True)
