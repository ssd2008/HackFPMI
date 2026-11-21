from functools import lru_cache
exec(open('.cache/round4/field/termexp.py').read().split("term_inputs=")[0])
from collections import defaultdict
from datetime import date
en=DenseIndex(docs,root/'.cache/ood-transfer-small-candidate/vectors/enron-mail')
vocab={w:i for i,w in enumerate(candidates)}
subject=[];body=[];boxes=defaultdict(list)
for i,d in enumerate(docs):
 text=d['doc'].lower();s=headers(d)[0];subject.append({vocab[w] for w in wordre.findall(s) if w in vocab});body.append({vocab[w] for w in wordre.findall(text[:2450]) if w in vocab});boxes[d['synthetic_user']].append(i)
qs={r['query_id']:r for r in map(json.loads,(suite/'queries_variants.jsonl').open())};meta={r['query_id']:r for r in map(json.loads,(suite/'cases.jsonl').open())}
samples=[]
for r in json.loads((private/'cos.json').read_bytes()):
 if r['ch']!='variants':continue
 q=qs[r['q']];m=meta[r['q']];qdate=q['query_date'];span=m.get('receipt_window');rows=[i for i in boxes[q['synthetic_user']] if not docs[i]['doc_dt'] or (docs[i]['doc_dt']<=qdate and (not span or span[0]<=docs[i]['doc_dt']<=span[1]))]
 samples.append((r['pure'],r['l'],rows))
ng=['пересадке редких комнатных кактусов','расписании чемпионата по подводным шахматам','скидке на противоблошиный шампунь','платье для танца фламенко','запасных деталях для марсохода','продаже париков для манекена','ремонте дирижабля в Антарктиде','рецепте оладий с облепихой']
for ids in boxes.values():
 if len(ids)>15:
  for query in ng:samples.append((query,{},ids))
stop=set('найди письмо письма мне нужны сообщения переписка переписку том теме упомянутых покажи было были обсуждении организации прошедших того который которым'.split())
@lru_cache(512)
def translation(phrase):
 xs=bank.vectors@bank.query(phrase)
 ix=xs.argpartition(-5)[-5:];return [(int(i),float(xs[i])) for i in ix]
results=[]
for num,(query,lab,inds) in enumerate(samples):
 terms=[t for t in re.findall('[а-яё]{3,}',query.lower()) if t not in stop]
 groups=[translation(' '.join(terms))]+[translation(t) for t in terms];rows=np.array(inds)
 cos=en.vectors[rows]@en.query(query);best=np.argsort(-cos)[:10]
 filters={}
 for floor in (.84,.85,.86,.87):
  lead=[[(t,z) for t,z in g if z>=floor] for g in groups]
  if not any(lead):continue
  tw={w for g in lead for w,z in g}
  scores=[];oks=[]
  for r,c in zip(rows,cos):
   if not body[r]&tw and not subject[r]&tw:
    oks.append(False);scores.append(float(c));continue
   support=[max((2 if t in subject[r] else 1 if t in body[r] else 0) for t,z in g) if g else 0 for g in lead]
   count=sum(z>0 for z in support)
   b=sum(min(.025,z*.0125) for z in support[:1])+sum(support[1:])* .018/max(1,len(terms))
   scores.append(float(c)+b)
   if len(terms)==1:
    ok=support[0]>=2 and support[1]>=2
   else:
    ok=all(z>0 for z in support[1:]) and (any(z>=2 for z in support[1:]) or support[0]>=2)
   oks.append(bool(ok and c>=.77))
  idx=np.flatnonzero(oks);match=idx[np.argsort(-np.array(scores)[idx])[:10]]
  ids=[docs[rows[j]]['doc_id'] for j in match]
  filters[str(floor)]=[bool(len(ids)), metrics(ids,lab,10)['ndcg'] if lab else 0]
 results.append(dict(pos=bool(lab),gates=filters))
for floor in (.84,.85,.86,.87):
 for pos in (True,False):
  z=[v['gates'].get(str(floor),[False,0]) for v in results if v['pos']==pos]
  print(floor,pos,'accepted',sum(t for t,_ in z),len(z),'ndcg',round(statistics.mean(v for t,v in z),3),flush=True)
