import sys,json,re,collections,statistics,pickle,os
from pathlib import Path
from datetime import date
import numpy as np
root=Path.cwd();private=root/'.cache/round4/field';suite=Path('/Users/kirill/Downloads/HackFPMI-mail-OOD/external/benchmarks/enron-mail')
sys.path.insert(0,str(root/'.cache/round4-mail-rank'))
from mailsearch.dense import DenseIndex
from mailsearch.fields import headers
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
sys.path.insert(0,str(root))
from benchmark import metrics
docs=[json.loads(l) for l in (suite/'documents.jsonl').open()]
wordre=re.compile(r"\b[a-z]{3,}(?:-[a-z]+)?\b")
df=collections.Counter();post=collections.defaultdict(list)
for i,d in enumerate(docs):
 title=headers(d)[0];text=re.sub(r'\b(?:[\w.!%-]+@[\w.-]+|https?://\S+)\b',' ',title)
 for t in set(wordre.findall(text.lower()))-ENGLISH_STOP_WORDS:
  df[t]+=1;post[t].append(i)
candidates=[t for t,n in sorted(df.items()) if n>=2 and n<len(docs)/8]
print('terms',len(candidates),flush=True)
bank=DenseIndex([{'doc_id':str(i),'doc':w} for i,w in enumerate(candidates)],private/'termbank')
term_inputs=list(dict.fromkeys(re.sub(r'^(?:мне нужны |покажи )','',v['pure'].lower()) for v in json.loads((private/'cos.json').read_bytes()) if v['ch']=='variants'))
print('sets',len(term_inputs))
for s in term_inputs+['кактусы','подводные шахматы','блошиный шампунь']:
 x=bank.vectors@bank.query(s);top=np.argsort(-x)[:3]
 print(s,[candidates[i] for i in top],np.round(x[top],3),flush=True)
