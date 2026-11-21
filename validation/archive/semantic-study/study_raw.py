from panels import W,H,D,HERE,get,jls
import sys,os,types,json,re
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
 if which=='rawgrid':
  t=(W/'mailsearch/retrieval.py').read_text();mark='        order = np.flatnonzero(plausible)\n'
  assert t.count(mark)==1
  t=t.replace(mark,'        self._trace(locals())\n'+mark)
  mod=types.ModuleType('mailsearch._semantic_study');mod.__package__='mailsearch';sys.modules[mod.__name__]=mod
  exec(compile(t,'study','exec'),mod.__dict__);engine=mod.SearchEngine(docs,cache)
 else:engine=src.SearchEngine(docs,cache)
 settings=['base','normal_08','raw_08','raw_15','raw_20','embedded_08','adaptive','adaptive_orig','adaptive_grad','rrf']
 summed={key:[] for key in settings} if which=='rawgrid' else {'base':[]}
 changed={key:0 for key in summed}; baseline_sets={}
 finding=re.compile(r'^\s*(?:(?:please\s+)?(?:find|look\s+for|search\s+for|show(?:\s+me)?|retrieve|list|locate)|найди(?:те)?|покажи(?:те)?|найти|отыщи(?:те)?|ищу)\b[\s,:-]*',re.I)
 wrapper=re.compile(r'^\s*(?:(?:my|the|all|any)\s+)*(?:(?:e-?mails?|messages?|letters?)\s+(?:about|regarding|concerning|related to)|(?:письм[оа-я]*|сообщени[еяй]+|уведомлени[еяй]+)\s+(?:о|об|про))\s+',re.I)
 question=re.compile(r'^(?:в\s+|на\s+|из\s+|с\s+|по\s+|под\s+|от\s+|для\s+|у\s+)?(?:как[а-яё]*|ч(?:то|ег[оа]|ем[у]?|ь[а-яё]+)|кт[оа]|ком[уы]?|когда|зачем|почему|скольк[а-яё]*|где|куда|котор[а-яё]*|what|whe(?:re|n)|who(?:se|m)?|which|how|why)\b', re.I)
 def trace(v):
  nonlocal alt
  if v['identifiers'] or v['phrases'] or len(v['rows'])<16:return
  raw=v['scope'].text if v['scope'].bounded else v['text'].strip()
  more=wrapper.sub('',finding.sub('',raw)).strip() if finding.match(raw) else raw
  vec=engine.dense.query(more) if more else v['vector'];cos2=engine.dense.vectors[v['rows']] @ vec
  rawvec=engine.dense.query(raw) if raw!=more else vec
  rawcos=engine.dense.vectors[v['rows']] @ rawvec if raw!=more else cos2
  c0,lex,pl=v['cosine'],v['lexical'],v['plausible'];ix=np.flatnonzero(pl)
  if not len(ix):return
  is_question=raw.rstrip().endswith('?') or bool(question.search(raw))
  w=.08 if is_question else .20
  # A dense query having a sharp contrast is useful even for declarative intents.
  c=cos2[ix]; k=min(20,len(c));head=np.partition(c,-k)[-k:]
  signal=float(head.max()-head.min()); wgrad=np.clip(.19-signal,.07,.19)
  for key,a in [('normal_08',c0+.08*lex),('raw_08',rawcos+.08*lex),('raw_15',rawcos+.15*lex),('raw_20',rawcos+.2*lex),('embedded_08',cos2+.08*lex),('adaptive',cos2+w*lex),('adaptive_orig',c0+w*lex),('adaptive_grad',cos2+wgrad*lex)]:
   selected=ix[np.argsort(-a[ix],kind='stable')[:10]]
   alt[key]=[engine.ids[v['rows'][i]] for i in selected]
 if which=='rawgrid':engine._trace=trace
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
