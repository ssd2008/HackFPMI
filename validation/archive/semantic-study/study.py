from panels import W,H,D,HERE,get,jls
import sys,os,types,json
from pathlib import Path
from datetime import date
from collections import defaultdict
import numpy as np
sys.path.insert(0,str(W))
from mailsearch.dense import _index_name
from mailsearch import retrieval as src
from statistics import fmean
import math
def metric(ids,lab):
 rel={d for d,g in lab.items() if g>0}
 if not rel:return [np.nan,np.nan,np.nan,int(bool(ids))]
 hits=[i for i,d in enumerate(ids,1) if d in rel]
 dcg=sum((2**lab.get(d,0)-1)/math.log2(i+1) for i,d in enumerate(ids,1))
 idcg=sum((2**g-1)/math.log2(i+1) for i,g in enumerate(sorted(lab.values(),reverse=True)[:10],1))
 return [dcg/idcg,len(hits)/len(rel),1/hits[0] if hits else 0,0]
def dataset(n):
 path=get(n);docs=jls(path/'documents.jsonl')
 cand=[H/'.cache/mailsearch',H/'.cache/ood-transfer-large/vectors'/n,H/'.cache/ood-transfer-small/vectors'/n]
 idx=_index_name(docs)
 cache=next((d for d in cand if (d/idx).is_file()),None)
 if cache is None:
  for file in (H/'.cache').glob('*/**/'+idx):
   if 'holdout' not in str(file) and 'round4-semantic' not in str(file):cache=file.parent;break
 if cache is None:
  cache=HERE/'vectors'/n;print('Building missing vectors',n,flush=True)
 qs={q['query_id']:q for q in jls(path/'queries_dev.jsonl')}
 if n in ('vk_dev','personal_dev'):qs.update({q['query_id']:q for q in jls(path/'queries_empty.jsonl')})
 labs=defaultdict(dict)
 for l in jls(path/'qrels_dev.jsonl'):labs[l['query_id']][l['doc_id']]=l['relevance']
 return docs,cache,qs,labs
def top(a,k):
 if len(a)<=k:return np.argsort(-a,kind='stable')
 return np.argpartition(-a,k-1)[:k]
def run(n,phase,which):
 docs,cache,qs,labs=dataset(n); ids=json.loads((HERE/'panel.json').read_bytes())[n][phase]
 if which=='grid':
  t=(W/'mailsearch/retrieval.py').read_text();mark='        order = np.flatnonzero(plausible)\n'
  assert t.count(mark)==1
  t=t.replace(mark,'        self._trace(locals())\n'+mark)
  mod=types.ModuleType('mailsearch._semantic_study');mod.__package__='mailsearch';sys.modules[mod.__name__]=mod
  exec(compile(t,'study','exec'),mod.__dict__);engine=mod.SearchEngine(docs,cache)
 else:engine=src.SearchEngine(docs,cache)
 settings=[f'w{v}p{p}' for v in (.0,.02,.05,.1,.2) for p in (.0,.01,.04)]+['rrf','rrf_dense','relative','base']
 summed={key:[] for key in settings} if which=='grid' else {'base':[]}
 changed={key:0 for key in summed}; baseline_sets={};stats=defaultdict(list)
 def trace(v):
  nonlocal alt
  if v['identifiers'] or v['phrases'] or len(v['rows'])<16:return
  cos,lex,pl=v['cosine'],v['lexical'],v['plausible']
  # Examine several fusion families with exactly the same already-accepted documents.
  # Query unknown/polarity constraints are therefore unchanged.
  ix=np.flatnonzero(pl)
  if not len(ix):return
  c=cos[ix]; l=lex[ix]; maxs=min(128,len(ix)); candidates=set(top(c,maxs))|set(top(l,maxs)); prelim={}
  for weight in (.0,.02,.05,.1,.2):
   a=c+weight*l; prelim[weight]=a; candidates.update(top(a,maxs))
  union=np.fromiter(sorted(candidates),dtype=np.int32)
  prox=engine.lexical.proximity(v['features'],v['rows'][ix[union]])
  def best(vals,subset=None):
   which=np.argsort(-vals,kind='stable')[:10]
   if subset is not None:which=subset[which]
   return [engine.ids[v['rows'][ix[i]]] for i in which]
  for w,a in prelim.items():
   for p in (.0,.01,.04):alt[f'w{w}p{p}']=best(a[union]+p*prox,union)
  for label,weight in [('rrf',1),('rrf_dense',2)]:
   dense=np.argsort(np.argsort(-c,kind='stable'),kind='stable');lexr=np.argsort(np.argsort(-l,kind='stable'),kind='stable')
   alt[label]=best(weight/(35+dense)+1/(35+lexr))
  cd=np.sort(c)[::-1];ld=np.sort(l)[::-1]; k=min(45,len(c)-1); weight=np.clip((cd[0]-cd[k])/(ld[0]-ld[k]+.001),.025,.25)
  alt['relative']=best(c+weight*l)
 if which=='grid':engine._trace=trace
 for i,qid in enumerate(ids):
  q=qs[qid];alt={}; base=[d for d,s in engine.search(q['synthetic_user'],q['query'],date.fromisoformat(q['query_date']),10)]; baseline_sets[qid]=base
  for key in summed:
   row=alt.get(key,base);summed[key].append(metric(row,labs[qid]));changed[key]+=row!=base
 def meanrows(a):
  a=np.asarray(a);ok=np.isfinite(a[:,0]);return [round(float(np.mean(a[ok,i])),6) if np.any(ok) else None for i in range(3)]+[round(float(np.mean(a[~ok,3])),6) if np.any(~ok) else None]
 report={k:dict(metrics=meanrows(v),changed=changed[k]) for k,v in summed.items()}
 (HERE/f'{which}-{phase}-{n}.json').write_text(json.dumps(report))
 if which!='grid':(HERE/f'{which}-{phase}-{n}-ranks.json').write_text(json.dumps(baseline_sets))
 print(n,which,phase,len(ids),report['base'],sorted([(v['metrics'][0],k) for k,v in report.items()],reverse=True)[:5],flush=True)
if __name__=='__main__':
 run(*sys.argv[1:])
