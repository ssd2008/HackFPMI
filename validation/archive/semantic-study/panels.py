import json,hashlib
from pathlib import Path
from collections import defaultdict
W=Path(__file__).resolve().parents[2]; H=W.parents[1]
D=Path('/Users/kirill/Downloads/HackFPMI-mail-OOD')
EXT=D/'external/benchmarks'; HERE=Path(__file__).resolve().parent
SALT='round4-semantic-v1'
def jls(p):
 with p.open() as f:return [json.loads(x) for x in f]
def hashed(v):return hashlib.sha256((SALT+'|'+v).encode()).hexdigest()
def get(n):
 if n=='public99':return H/'data'
 if n in ('vk_dev','personal_dev'):return D/'suite'/n
 return EXT/n
def selection(n):
 root=get(n); qs=jls(root/'queries_dev.jsonl'); labs=jls(root/'qrels_dev.jsonl');groups={}
 if n in ('vk_dev','personal_dev','enron-mail'):
  cases={r['query_id']:r for r in jls(root/'cases.jsonl')}
  if n=='enron-mail':qs=[q for q in qs if cases[q['query_id']]['judgment']=='semantic_silver']
  if n!='enron-mail':qs+=jls(root/'queries_empty.jsonl')
  groups={q['query_id']:str(cases[q['query_id']]['family_id']) for q in qs}
 elif n=='rubq-2.0':
  old=json.loads((H/'.cache/rubq-transfer/clusters.json').read_bytes())
  groups={q['query_id']:old[q['query_id'].split('/')[-1]] for q in qs}
 elif n!='public99':
  d=defaultdict(set)
  for l in labs:
   if l['relevance']>0:d[l['query_id']].add(l['doc_id'].split('#')[0])
  inv=defaultdict(set)
  for qid,ds in d.items():
   for doc in ds:inv[doc].add(qid)
  for first in sorted(d):
   if first in groups:continue
   stack=[first]
   while stack:
    i=stack.pop()
    if i in groups:continue
    groups[i]=first
    for doc in d[i]:stack.extend(inv.pop(doc,()))
 else:groups={q['query_id']:q['query_id'] for q in qs}
 splits=defaultdict(list)
 for q in qs:
  qid=q['query_id']; grp=groups[qid]; k=hashed(n+'|group|'+grp)
  phase='explore' if int(k[:8],16)%2==0 else 'confirm'
  splits[phase].append(qid)
 for phase in ('explore','confirm'):
  splits[phase]=sorted(splits[phase],key=lambda i:hashed(n+'|'+i))[:120]
 if n in ('enron-mail','public99'):splits['explore']=[]
 return dict(splits,full=[q['query_id'] for q in qs])
if __name__=='__main__':
 out={n:selection(n) for n in ('mr-tydi-ru','miracl-ru','rubq-2.0','beir-scifact','enron-mail','personal_dev','vk_dev','public99')}
 p=HERE/'panel.json';data=json.dumps(out,sort_keys=True).encode()
 if p.exists():assert p.read_bytes()==data
 else:p.write_bytes(data)
 print('Frozen panels',hashlib.sha256(data).hexdigest(),{n:(len(v['explore']),len(v['confirm']),len(v['full'])) for n,v in out.items()})
