from __future__ import annotations
from collections import Counter,defaultdict
from functools import lru_cache
from pathlib import Path
import gzip, json,re,sys,statistics
import numpy as np
import snowballstemmer
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
root=Path.cwd();p=root/'.cache/round4/field';ds=Path('/Users/kirill/Downloads/HackFPMI-mail-OOD/external/benchmarks/enron-mail')
sys.path.insert(0,str(root/'.cache/round4-mail-rank'))
from mailsearch.dense import DenseIndex
from mailsearch.fields import headers
from mailsearch.retrieval import _stem,_STOP,_REQUEST_WORDS
from benchmark import metrics
data=lambda file:[json.loads(l) for l in file.open()]
eng=snowballstemmer.stemmer('english');estem=lru_cache(maxsize=80000)(eng.stemWord)
lex={}
with gzip.open(p/'lex-eng-rus.tsv.gz','rt') as source:
 for line in source:
  k,subs=line.rstrip('\n').split('\t');lex[k]=subs.split('|')
docs=data(ds/'documents.jsonl')
de=DenseIndex(docs,root/'.cache/ood-transfer-small-candidate/vectors/enron-mail')
m={r['query_id']:r for r in data(ds/'cases.jsonl')};qs={r['query_id']:r for r in data(ds/'queries_variants.jsonl')}
# Sparse English postings, separately for subject and body, with opacity-preserving token boundaries.
indices=[];weights=[];indptr=[0];vocabulary={};boxes=defaultdict(list)
english=re.compile(r"[a-z]{3,}")
for i,d in enumerate(docs):
 boxes[d['synthetic_user']].append(i)
 sub=headers(d)[0];text=d['doc'][:3400].lower()
 body=Counter(estem(t) for t in english.findall(text) if t not in ENGLISH_STOP_WORDS)
 for w in set(english.findall(sub))-ENGLISH_STOP_WORDS:body[estem(w)]+=4
 for token,w in body.items():
  indices.append(vocabulary.setdefault(token,len(vocabulary)));weights.append(min(w,20))
 indptr.append(len(indices))
matrix=csr_matrix((np.asarray(weights,np.float32),indices,indptr),shape=(len(docs),len(vocabulary))).tocsc()
freq=np.diff(matrix.indptr)
idf=np.log(1+(len(docs)-freq+.5)/(freq+.5))
stop={_stem(w) for w in ('которые котором которых переписка переписку упомянутых теме содержат тех было были получены обсуждают хочу прочитать найти').split()}|_REQUEST_WORDS
def concepts(text):
 words=re.findall('[а-яё]{2,}',text.casefold().replace('ё','е')); groups=[];altnext=False;unknown=[]
 for w in words:
  if w in ('или','либо'):
   altnext=True;continue
  stem=_stem(w)
  if w in _STOP or stem in stop or len(w)<3:continue
  raw=lex.get(stem,[])
  if not raw:
   # A participle can modify the translated object. Its disappearance should not turn a genuine unknown noun into a match.
   participle=bool(re.search(r'(?:нн|ющ|вш|ющ|ащ|ящ|ем|им)[аеиоыу][еяйхмуюи]?',w))
   if not participle:unknown.append(w)
   continue
  vals=[]
  for sense,x in enumerate(raw[:15]):
   terms=[vocabulary.get(estem(en)) for en in english.findall(x) if en not in ENGLISH_STOP_WORDS]
   if terms and all(t is not None for t in terms): vals.append((terms,sense))
  if altnext and groups:groups[-1].extend(vals)
  else:groups.append(vals)
  altnext=False
 return groups,unknown
def score(query,rows,ranksource):
 groups,unk=concepts(query)
 if not groups or unk or any(not g for g in groups):return []
 scores=[];matches=[]
 for choices in groups:
  score=np.zeros(len(rows)); hit=np.zeros(len(rows),bool)
  for terms,sense in choices:
   strength=np.min(np.vstack([matrix[rows,t].toarray().ravel() for t in terms]),axis=0)
   hit|=strength>0
   score=np.maximum(score,np.log1p(strength)*float(np.mean(idf[terms]))/(1+.06*sense))
  scores.append(score); matches.append(hit)
 eligible=np.flatnonzero(np.logical_and.reduce(matches))
 if not len(eligible):return []
 a=np.mean(scores,axis=0)
 cos=de.vectors[rows[eligible]]@de.query(query)
 order=np.argsort(-(cos+.023*np.log1p(a[eligible])),kind='stable')
 return [(int(rows[eligible[j]]),float(cos[j]),float(a[eligible[j]])) for j in order[:30]]
raw=json.loads((p/'cos.json').read_bytes());dataset=[]
for item in raw:
 if item['ch']!='variants':continue
 q=qs[item['q']];meta=m[item['q']];span=meta.get('receipt_window');rows=np.asarray([i for i in boxes[q['synthetic_user']] if (not docs[i]['doc_dt'] or docs[i]['doc_dt']<=q['query_date']) and (not span or docs[i]['doc_dt'] and span[0]<=docs[i]['doc_dt']<=span[1])])
 dataset.append((item['pure'],rows,item['l'],item['q']))
bad=[('хамелеонах',[r'chameleon']),('лекциях о терменвоксе',[r'theremin']),('струнах для балалайки',[r'balalaika']),('дрессировке фламинго',[r'flamingo']),('шахматах под водой',[r'chess',r'underwater|scuba']),('отчетах марсоходов',[r'mars rover|martian rover']),('билетах на бои единорогов',[r'unicorn']),('кактусах',[r'cact(us|i)']),('покупке дирижабля',[r'zeppelin|dirigible|airship|blimp']),('ремонте ксилофона',[r'xylophone']),('выставке мамонтов',[r'mammoth']),('оплате доставки телескопа',[r'telescope']),('ювелирной гравировке',[r'engraving|engraver'])]
for query,patterns in bad:
 for ix in boxes.values():
  if len(ix)<15:continue
  if any(all(re.search(pat,docs[i]['doc'],re.I) for pat in patterns) for i in ix):continue
  dataset.append((query,np.asarray(ix),{},''))
found=[]
for query,rows,lab,qid in dataset:
 hit=score(query,rows,lab)
 found.append({'q':qid,'query':query,'positive':bool(lab),'hits':hit,'l':lab})
(p/'dict-trial.json').write_text(json.dumps(found))
for cutoff in (0,.76,.78,.8):
 for pos in (True,False):
  obs=[r for r in found if r['positive']==pos];lists=[[docs[i]['doc_id'] for i,c,a in r['hits'] if c>=cutoff][:10] for r in obs]
  print(cutoff,pos,len(obs),sum(map(bool,lists)),round(statistics.mean(metrics(ls,r['l'],10)['ndcg'] for ls,r in zip(lists,obs)),4) if pos else dict(Counter(r['query'] for r,ls in zip(obs,lists) if ls)),flush=True)
