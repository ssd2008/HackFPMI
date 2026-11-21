import sys,json,re,collections,statistics
from pathlib import Path
import numpy as np
root=Path.cwd();private=root/'.cache/round4/field';s=Path('/Users/kirill/Downloads/HackFPMI-mail-OOD/external/benchmarks/enron-mail')
sys.path.insert(0,str(root/'.cache/round4-mail-rank'))
from mailsearch.dense import DenseIndex
from mailsearch.fields import headers
from benchmark import metrics
load=lambda p: [json.loads(l) for l in p.open()]
docs=load(s/'documents.jsonl');en=DenseIndex(docs,root/'.cache/ood-transfer-small-candidate/vectors/enron-mail')
bybox=collections.defaultdict(list);vocab={};mapping=[]
for i,d in enumerate(docs):
 title=headers(d)[0];bybox[d['synthetic_user']].append(i)
 mapping.append(vocab.setdefault(title,len(vocab)))
print('subjects',len(vocab),flush=True)
sub=DenseIndex([dict(doc_id=str(i),doc=t) for t,i in vocab.items()],private/'topics')
mapping=np.array(mapping)
m={q['query_id']:q for q in load(s/'cases.jsonl')}; qs={q['query_id']:q for ch in ('dev','variants') for q in load(s/f'queries_{ch}.jsonl')}
ref={r['q']:r for r in json.loads((private/'cos.json').read_bytes()) if 'q' in r}
ovec=[]
neg=['пересадке редких комнатных кактусов','расписании чемпионата по подводным шахматам','скидке на противоблошиный шампунь','платье для танца фламенко','запасных деталях для марсохода','продаже париков для манекена','ремонте дирижабля в Антарктиде','рецепте оладий с облепихой']
for i,(qid,r,dids,query) in enumerate([(qid,v,
 [i for i in bybox[qs[qid]['synthetic_user']] if docs[i]['doc_dt'] and docs[i]['doc_dt']<=qs[qid]['query_date'] and (not m[qid].get('receipt_window') or m[qid]['receipt_window'][0]<=docs[i]['doc_dt']<=m[qid]['receipt_window'][1])],v['pure']) for qid,v in ref.items()]+[('',{'ch':'neg'},ids,query) for ids in list(bybox.values())[:25] if len(ids)>=15 for query in neg]):
 rows=np.array(dids);vec=en.query(query)
 a=en.vectors[rows]@vec;b=sub.vectors[mapping[rows]]@vec
 x=.35*a+.65*b
 if qid:
  known=r['l']
  def measure(v):return metrics([docs[j]['doc_id'] for j in rows[np.argsort(-v)[:10]]],known,10)['ndcg']
 else:measure=lambda v:None
 ovec.append(dict(channel=r['ch'],n=len(rows),ca=float(a.max()),cb=float(b.max()),cx=float(x.max()),raw=measure(a),sub=measure(b),mix=measure(x),
                   soft=measure(x) if x.max()>=.80 else (0 if qid else None)))
(private/'subject-scores.json').write_text(json.dumps(ovec))
for ch in ('dev','variants','neg'):
 v=[z for z in ovec if z['channel']==ch]
 print(ch,len(v),[(k,np.round(np.quantile([z[k] for z in v],[.1,.5,.9]),3)) for k in ('ca','cb','cx')], [(k,round(statistics.mean(z[k] for z in v),3)) for k in ('raw','sub','mix','soft')] if ch!='neg' else '')
