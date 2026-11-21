from datetime import date
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import importlib.util,json,os,subprocess,sys,types
ROOT=Path.cwd()
sys.path[:0]=[str(ROOT/'.cache/round4/prior-search'/os.getenv('PRIOR_CHECK_REV','9a720b79d78ab7a3012c070833144c46a34786db')/'engine'),str(ROOT/'tools')]
import mailsearch.retrieval as ret
from check_ood_transfer import scores
from statistics import fmean
rows=lambda f:[json.loads(l) for l in Path(f).read_bytes().splitlines()]
docs=rows('data/documents.jsonl'); e=ret.SearchEngine(docs,ROOT/'.cache/mailsearch')
old=types.ModuleType('old_date');sys.modules[old.__name__]=old
exec(subprocess.check_output(['git','show','a211900:mailsearch/dates.py']),old.__dict__)
queries=rows('tests/fixtures/validation_pilot/queries.jsonl')
tasks={r['query_id']:r for r in rows('tests/fixtures/validation_pilot/tasks.jsonl')}
labels=defaultdict(dict)
for r in rows('tests/fixtures/validation_pilot/qrels.jsonl'):labels[r['query_id']][r['doc_id']]=r['relevance']
cases=[]
for q in queries:
 i=q['query_id'];t=tasks[i];cases.append((q['query'],q['synthetic_user'],q['query_date'],labels[i],t['split']))
for name in ('search_probes','search_probes_extra'):
 for q in json.loads((ROOT/'tests/fixtures'/(name+'.json')).read_bytes())['cases']:
  cases.append((q['query'],q['mailbox'],q['date'],dict.fromkeys(q['expected'],1),name))
orig_date=ret.parse_date_scope; orig_scorer=ret.semantic_lexical_scale
for ds in (('present',) if os.getenv('ONE_VERSION') else ('previous','present')):
 for sc in (('present',) if os.getenv('ONE_VERSION') else ('previous','present')):
  ret.parse_date_scope=old.parse_date_scope if ds=='previous' else orig_date
  ret.semantic_lexical_scale=(lambda q:None) if sc=='previous' else orig_scorer
  values=defaultdict(list)
  for query,box,day,gold,group in cases:
   rank=[d for d,s in e.search(box,query,date.fromisoformat(day),10)]; s=scores(rank,gold); values[group].append(s|{'hit5':bool(set(rank[:5])&gold.keys()) if gold else False})
  output={}
  for group,ls in values.items():
   pos=[x for x in ls if 'ndcg_at_10' in x]; output[group]=[round(fmean(x[k] for x in pos),6) for k in ['ndcg_at_10','known_recall_at_10','mrr_at_10']]+[sum(x['hit5']for x in pos)]
  print(ds,sc,output,flush=True)
