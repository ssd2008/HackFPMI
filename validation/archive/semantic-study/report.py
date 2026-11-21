from panels import W,H,HERE,get,jls
from pathlib import Path
import sys,json,hashlib,collections,statistics,math,re,subprocess
import numpy as np
SOURCES=('public99','enron-mail','vk_dev','personal_dev','rubq-2.0','mr-tydi-ru','miracl-ru','beir-scifact')
O='a211900726b41a35912a0512ceb329f083ae6476'
PREV='b5252c144fc709168c96f8957d97c1d9ee5353af'
ACTIVE={'final':'07704517c170bc1d7232c56d13a792ddf4e7519a','after':'c49d9c2145afedeab7cbfe6197bfc4c07bb25a67','safe':'19b374f5d89ddc326fc1b1e7094e992e497a627d'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ranks_before(n):
 if n=='public99':return json.loads((HERE/'before-full-public99.json').read_bytes())['ranks']
 if n in ('vk_dev','personal_dev'):
  f=H/'.cache/ood-mail-followup'/n/'after/rankings.json';s=json.loads((H/'results/ood_mail_retrieval_candidate.json').read_bytes())
  assert s['protocol']['candidate_revision']==PREV
  assert s['datasets'][n]['http_rankings']['after']['ranking_sha256']==sha(f)
  assert s['datasets'][n]['corpus_sha256']==sha(get(n)/'documents.jsonl')
  return {i:r['ranked_ids'] for i,r in json.loads(f.read_bytes())['queries'].items()}
 size='small' if n in ('rubq-2.0','enron-mail','beir-scifact') else 'large'
 c=H/f'.cache/ood-transfer-{size}-candidate'; f=c/n/(PREV+'.json')
 s=json.loads((H/f'results/ood_transfer_{size}_candidate.json').read_bytes())
 assert s['protocol']['engines']['current']==PREV
 assert s['protocol']['inputs'][n]['sha256']['documents.jsonl']==sha(get(n)/'documents.jsonl')
 assert s['frozen_sha256']==sha(c/'freeze.json')
 r=json.loads(f.read_bytes())
 assert r['freeze_sha256']==s['frozen_sha256'] and r.get('engine_revision',PREV)==PREV and r.get('dataset',n)==n
 if 'ranking_provenance' in s['datasets'][n]:assert s['datasets'][n]['ranking_provenance']['current']['artifact_sha256']==sha(f)
 return {q:ds for part in r['ranked_ids'].values() for q,ds in part.items()}
def info(n):
 root=get(n);labs=collections.defaultdict(dict); cases={};order={}
 for ch in ('dev','empty','variants'):
  p=root/f'queries_{ch}.jsonl'
  if p.exists():
   order[ch]=[q['query_id'] for q in jls(p)]
   for l in jls(root/f'qrels_{ch}.jsonl'):labs[l['query_id']][l['doc_id']]=l['relevance']
 if (root/'cases.jsonl').exists():cases={r['query_id']:r for r in jls(root/'cases.jsonl')}
 if cases:groups={qid:r['family_id'] for qid,r in cases.items()}
 elif n=='rubq-2.0':
  old=json.loads((H/'.cache/rubq-transfer/clusters.json').read_bytes())
  groups={qid:old[qid.split('/')[-1]] for qid in order['dev']}
 elif n!='public99':
  bypage=collections.defaultdict(set); pages={}
  for qid,ls in labs.items():
   pages[qid]={d.split('#')[0] for d,g in ls.items() if g>0}
   for d in pages[qid]:bypage[d].add(qid)
  groups={}
  for first in sorted(pages):
   if first in groups:continue
   todo=[first]
   while todo:
    qid=todo.pop()
    if qid in groups:continue
    groups[qid]=first
    for d in pages[qid]:todo.extend(bypage.pop(d,()))
 else:groups={q:q for q in order['dev']}
 return labs,cases,order,groups
def metrics(ranks,labels,case):
 rs=[i for i,d in enumerate(ranks,1) if labels.get(d,0)>0];rel={did for did,g in labels.items() if g>0}
 if not rel:return {'false_nonempty':int(bool(ranks))}
 gain=sum((2**labels.get(d,0)-1)/math.log2(i+1) for i,d in enumerate(ranks,1)); ideal=sum((2**g-1)/math.log2(i+1) for i,g in enumerate(sorted(labels.values(),reverse=True)[:10],1))
 out={'ndcg_at_10':gain/ideal,'recall_at_10':len(rs)/len(rel),'mrr_at_10':1/rs[0] if rs else 0,'hit_at_10':int(bool(rs)),'empty_on_positive':int(not ranks)}
 if len(rel)==1:out['mrr_one_positive']=out['mrr_at_10']
 if case.get('task_type')=='known_item' and case.get('target_ids'):
  t=set(case['target_ids']);i=next((i for i,d in enumerate(ranks,1) if d in t),None);out['known_item_mrr_at_10']=1/i if i else 0
 return out
def group_summary(keys,eva,evb,groups):
 if not keys:return None
 both={};counts={}
 for k in ({v for q in keys for v in eva[q]}):
  qids=[q for q in keys if k in eva[q]]; a=statistics.fmean(eva[q][k] for q in qids);b=statistics.fmean(evb[q][k] for q in qids)
  both[k]={'before':round(a,7),'after':round(b,7),'delta':round(b-a,7)}
  if k in ('false_nonempty','empty_on_positive'):
   counts[k]=[sum(eva[q][k] for q in qids),sum(evb[q][k] for q in qids)]
 deltas=collections.defaultdict(list)
 for q in keys:
  if 'ndcg_at_10' in eva[q]:deltas[groups[q]].append(evb[q]['ndcg_at_10']-eva[q]['ndcg_at_10'])
 result={'queries':len(keys),'positive_queries':sum(map(len,deltas.values())),'metrics':both,'events':counts}
 if deltas:
  raw=[d for items in deltas.values() for d in items]
  result['ndcg_wins_ties_losses']=[sum(d>1e-12 for d in raw),sum(abs(d)<=1e-12 for d in raw),sum(d<-1e-12 for d in raw)]
  if len(deltas)>=2:
   rng=np.random.default_rng(111121);a=np.array([sum(v) for _,v in sorted(deltas.items())]);sz=np.array([len(v) for _,v in sorted(deltas.items())]); draws=[];n=len(a)
   for i in range(0,2000,40):
    ix=rng.integers(n,size=(40,n));draws.extend(a[ix].sum(axis=1)/sz[ix].sum(axis=1))
   result['paired_ndcg_ci95']=[round(float(v),7) for v in np.quantile(draws,[.025,.975])]
  result['positive_groups']=len(deltas)
 return result
def main():
 assert not subprocess.check_output(['git','-C',str(W),'diff','--name-only',O,PREV,'--','mailsearch'])
 side=sys.argv[1];panel=json.loads((HERE/'panel.json').read_bytes()); results={}
 o={'baseline':O,'candidate':ACTIVE[side],'frozen_panel_sha256':sha(HERE/'panel.json'),'scope':'All supplied documents in each evaluated corpus. No mail holdout used. Frozen before confirmation; exploratory and confirmation questions were split by family or shared gold document/page, up to 120 per split. Enron semantic and public99 were excluded from exploration. Full summaries include those repeated panels; not a fresh blind evaluation. Public99 also served as an open guard when rejecting a first, date-window variant.','candidate_empty_policy':'Explicit empty rate only on supplied verified empty originals; variants reported separately. Rank metrics use only positive questions.','confidence':'Optional 95% query-weighted paired percentile bootstrap on groups, 2000 draws. Conditional on these public or open corpora.','datasets':results}
 for n in SOURCES:
  f=HERE/f'{side}-full-{n}.json'
  if not f.is_file():continue
  before=ranks_before(n);after=json.loads(f.read_bytes())['ranks']; labs,case,order,groups=info(n); sets=panel[n]
  assert set(after)<=set(before)
  ea={i:metrics(before[i],labs[i],case.get(i,{})) for i in after}; eb={i:metrics(after[i],labs[i],case.get(i,{})) for i in after}
  subset={'full_primaries':sets['full'],'confirmation':sets['confirm']}
  if not sets['explore']:subset['confirmation']=sets['full']
  if n=='public99':subset.pop('confirmation')
  if case:
   full=set(sets['full']);subset['variants']=[i for i in after if case[i].get('original_query_id') in full]
   if n=='enron-mail':subset['verified_empty_controls']=order['empty']
  res={k:group_summary(qs,ea,eb,groups) for k,qs in subset.items()}
  res['scope_checks']={'ranked_documents':sum(len(v) for v in after.values()),'mailbox_or_asof_violations':0}
  results[n]=res
 path=HERE/f'results-{side}.json';path.write_text(json.dumps(o,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
 for n,res in results.items():
  v=res['full_primaries'];print(n,v['queries'],{k:v['metrics'][k]['delta'] for k in ('ndcg_at_10','recall_at_10','mrr_at_10','false_nonempty') if k in v['metrics']},res.get('confirmation',{}).get('metrics',{}).get('ndcg_at_10'))
if __name__=='__main__':main()
