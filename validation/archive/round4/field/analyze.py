import json,pathlib, sys, collections, statistics
root=pathlib.Path.cwd();private=root/'.cache/round4/field';pool=pathlib.Path('/Users/kirill/Downloads/HackFPMI-mail-OOD')
sys.path.insert(0,str(root))
from benchmark import metrics as rank_metrics
def load(p):return [json.loads(l) for l in p.open()]
def summary(rows):
 def avg(k):return round(statistics.mean(r[k] for r in rows if k in r),5) if any(k in r for r in rows) else None
 return {'n':len(rows),**{k:avg(k) for k in ('ndcg','recall','mrr_known','recall_ceiling')},'false_empty':sum(r.get('false_empty',0) for r in rows),'false_nonempty':sum(r.get('false_nonempty',0) for r in rows)}
def aggregate(ds,ranks):
 meta={x['query_id']:x for x in load(ds/'cases.jsonl')};rels={}
 for ch in ('dev','empty','variants'):
  for qrel in load(ds/f'qrels_{ch}.jsonl'):
   rels.setdefault(qrel['query_id'],{})[qrel['doc_id']]=qrel['relevance']
 by=collections.defaultdict(list)
 for qid,ids in ranks.items():
  m=meta[qid]; group=m.get('variant','original') if m['state']=='positive' else 'empty';known=rels.get(qid,{})
  if m['state']=='positive':
   o=rank_metrics(ids,known,10);n=sum(g>0 for g in known.values());o.update(false_empty=not ids,recall_ceiling=min(10,n)/n)
   if m['task_type']=='known_item':
    tar=set(m['target_ids']);o['mrr_known']=next((1/i for i,d in enumerate(ids,1) if d in tar),0.)
  else:o={'false_nonempty':bool(ids)}
  by[group].append(o);by[group+':'+m['family']].append(o)
 return {k:summary(v) for k,v in sorted(by.items())}
if __name__=='__main__':
 out={}
 for name in ('vk_dev','personal_dev','enron-mail'):
  suites=pool/('external/benchmarks' if name=='enron-mail' else 'suite')/name
  results={}
  for side in ['before','after']:
   if name=='enron-mail':
    file=(root/'.cache/ood-transfer-small-candidate/enron-mail/b5252c144fc709168c96f8957d97c1d9ee5353af.json') if side=='before' else private/f'enron-{sys.argv[2]}.json'
    rank={q:ids for g in json.loads(file.read_bytes())['ranked_ids'].values() for q,ids in g.items()}
   else:
    file=root/'.cache/ood-mail-followup'/name/'after'/'rankings.json' if side=='before' else private/sys.argv[1]/name/'rankings.json'
    rank={q:row['ranked_ids'] for q,row in json.loads(file.read_bytes())['queries'].items()}
   results[side]=aggregate(suites,rank)
  out[name]=results
  for ch in ('original','paraphrase','typo','empty'):
   if ch in results['before']:
    a,b=results['before'][ch],results['after'][ch];print(name,ch,[str(a[k])+'→'+str(b[k]) for k in ('ndcg','recall','mrr_known','false_empty','false_nonempty')])
 (private/f'quality-{sys.argv[1]}.json').write_text(json.dumps(out,indent=2))
