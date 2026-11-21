import sys,json,re,collections,statistics
from pathlib import Path
from datetime import date
import numpy as np
root=Path.cwd();private=root/'.cache/round4/field';s=Path('/Users/kirill/Downloads/HackFPMI-mail-OOD/external/benchmarks/enron-mail')
sys.path.insert(0,str(root/'.cache/round4-mail-rank'))
from mailsearch.dense import DenseIndex
from benchmark import metrics
load=lambda p:[json.loads(l) for l in p.open()]
docs=load(s/'documents.jsonl'); index=DenseIndex(docs,root/'.cache/ood-transfer-small-candidate/vectors/enron-mail')
boxes=collections.defaultdict(list)
for i,d in enumerate(docs):boxes[d['synthetic_user']].append(i)
meta={m['query_id']:m for m in load(s/'cases.jsonl')}
out=[]
prefix=re.compile(r'^(?:(?:find|show|search for)\s+(?:(?:me|all|the)\s+)*(?:e?mails?|messages?)\s+(?:about|concerning|regarding|on)\s+|(?:покаж[иите]*|найд[итеи]*|нужн[аы]|отыщи|(?:мне\s+)?нужны)\s+(?:переписку|письма|сообщения|уведомления)\s+(?:об?|с|про)\s+)',re.I)
def evaluate(ch):
 labs=collections.defaultdict(dict)
 for l in load(s/f'qrels_{ch}.jsonl'):labs[l['query_id']][l['doc_id']]=l['relevance']
 for q in load(s/f'queries_{ch}.jsonl'):
  m=meta[q['query_id']]; lab=labs[q['query_id']]
  if not m['family'].startswith('semantic'):continue
  span=m.get('receipt_window')
  rows=np.array([i for i in boxes[q['synthetic_user']] if not docs[i]['doc_dt'] or docs[i]['doc_dt']<=q['query_date']])
  if span:rows=np.array([i for i in rows if docs[i]['doc_dt'] and span[0]<=docs[i]['doc_dt']<=span[1]])
  text=re.split(r'\breceived\b|[,]?\s+полученные\b',q['query'],maxsplit=1)[0].strip(' .?')
  query=prefix.sub('',text)
  cos=index.vectors[rows]@index.query(query)
  order=np.argsort(-cos)[:10];found=[docs[rows[j]]['doc_id'] for j in order]
  out.append(dict(ch=ch,n=len(rows),q=q['query_id'],pure=query,cos=cos[order].tolist(),bg=np.quantile(cos,[.25,.5,.75,.95]).tolist(),r=found,l=lab))
for ch in ('dev','variants'):evaluate(ch)
negatives=['пересадке редких комнатных кактусов','расписании чемпионата по подводным шахматам','скидке на противоблошиный шампунь','платье для танца фламенко','запасных деталях для марсохода','продаже париков для манекена','ремонте дирижабля в Антарктиде','рецепте оладий с облепихой']
for text in negatives:
 vec=index.query(text)
 for box,rs in list(boxes.items())[:25]:
  if len(rs)<15:continue
  rows=np.array(rs);cos=index.vectors[rows]@vec
  so=np.argsort(-cos)[:10]
  out.append(dict(ch='neg',n=len(rs),cos=cos[so].tolist(),bg=np.quantile(cos,[.25,.5,.75,.95]).tolist()))
(private/'cos.json').write_text(json.dumps(out,ensure_ascii=False))
for ch in ['dev','variants','neg']:
 rs=[r for r in out if r['ch']==ch]
 for kind,rr in [('all',rs),('small',[r for r in rs if r['n']<16]),('large',[r for r in rs if r['n']>=16])]:
  if not rr:continue
  print(ch,kind,len(rr), 'max',np.round(np.quantile([r['cos'][0] for r in rr],[.1,.5,.9]),3),'gap',np.round(np.quantile([r['cos'][0]-r['bg'][1] for r in rr],[.1,.5,.9]),3))
  if ch!='neg':print('dense all',round(statistics.mean(metrics(r['r'],r['l'],10)['ndcg'] for r in rr),3))
