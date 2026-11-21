import json, re, sys, collections, time, statistics, random, ast
from pathlib import Path
from datetime import date
import numpy as np
root = Path.cwd(); work = root/'.cache/round4-mail-rank'; scratch=root/'.cache/round4/field'
sys.path.insert(0,str(work))
from mailsearch.retrieval import SearchEngine,_stem,_STOP,_REQUEST_WORDS
import importlib.util,mailsearch.retrieval
spec=importlib.util.spec_from_file_location("round_dates",root/"mailsearch/dates.py");month=importlib.util.module_from_spec(spec);sys.modules[spec.name]=month;spec.loader.exec_module(month);mailsearch.retrieval.parse_date_scope=month.parse_date_scope
dataset=Path('/Users/kirill/Downloads/HackFPMI-mail-OOD/external/benchmarks/enron-mail')
read=lambda fp:[json.loads(l) for l in fp.open()]
docs=read(dataset/'documents.jsonl')
e=SearchEngine(docs,root/'.cache/ood-transfer-small-candidate/vectors/enron-mail')
bad=ast.literal_eval(next(line[4:] for line in (scratch/'evaldict.py').read_text().splitlines() if line.startswith('bad=')))
base=e.bilingual.search
out={}
for suffix in ('','invoice','meeting','budget'):
 counts=collections.Counter()
 for query,patterns in bad:
  for box,rows in e.boxes.items():
   if len(rows)<15 or any(all(re.search(pat,docs[i]['doc'],re.I) for pat in patterns) for i in rows):continue
   words='Покажи письма о '+query+' '+suffix; d=date(2100,1,1)
   e.bilingual.search=lambda *a:[]
   b=e.search(box,words,d)
   e.bilingual.search=base
   a=e.search(box,words,d)
   counts['n']+=1;counts['base']+=bool(b);counts['after']+=bool(a);counts['added']+=bool(a and not b)
 out[suffix or 'ru']=dict(counts)
print('negatives',out,flush=True)
samples=[]
for q in read(dataset/'queries_variants.jsonl'):
 if re.fullmatch('[а-яёА-ЯЁ0-9 .:,?—–-]+',q['query']):
  samples.append(q)
samples=[q for q in samples if q['query_id'] in {v['q'] for v in json.loads((scratch/'cos.json').read_bytes()) if v['ch']=='variants'}]
random.Random(2026).shuffle(samples);negtime={True:[], False:[]}
for i,q in enumerate(samples):
 d=date.fromisoformat(q['query_date'])
 for enabled in (bool(i%2),not i%2):
  e.bilingual.search=base if enabled else (lambda *a:[])
  start=time.perf_counter();e.search(q['synthetic_user'],q['query'],d);negtime[enabled].append((time.perf_counter()-start)*1e3)
quant=lambda xs:[round(float(x),3) for x in np.quantile(xs,[.5,.95,.99])]
print('topic queries',len(samples),'baseline and current median/p95/p99',quant(negtime[False]),quant(negtime[True]),flush=True)
(scratch/'cross-guard-result.json').write_text(json.dumps(dict(negative=out,topicqueries=len(samples),baseline_ms=quant(negtime[False]),after_ms=quant(negtime[True]))))
