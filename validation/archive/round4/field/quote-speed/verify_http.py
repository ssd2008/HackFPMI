from pathlib import Path
import json, sys
from concurrent.futures import ThreadPoolExecutor
root=Path.cwd();scratch=root/'.cache/round4/field/quote-speed';sys.path.insert(0,str(root));from benchmark import fetch
read=lambda path:[json.loads(l) for l in path.open()]
out={}
for split,port in [('vk_dev',18861),('vk_holdout',18861),('personal_dev',18862),('personal_holdout',18862)]:
 ds=root/'.cache/round4/final'/split/'after';queries=read(ds/'queries.jsonl');prev=json.loads((ds/'rankings.json').read_bytes())['queries']
 def job(q):
  ranks,_,_=fetch(f'http://127.0.0.1:{port}',q,10)
  return list(ranks)==prev[q['query_id']]['ranked_ids']
 with ThreadPoolExecutor(4) as jobs:changed=sum(not x for x in jobs.map(job,queries))
 out[split]={'queries':len(queries),'changed':changed};print(split,out[split],flush=True)
(scratch/'full-rank-equivalence.json').write_text(json.dumps(out,indent=2))
