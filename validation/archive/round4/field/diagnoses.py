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
review=e.bilingual.ungrounded
from mailsearch.bilingual import _SELECT,_UNSUPPORTED,_CYRILLIC,_BOILERPLATE,_ENGLISH,_BODY
reasons=collections.Counter()
changed=collections.Counter()
for suffix in ('','invoice','meeting','budget'):
 for query,patterns in bad:
  for box,rows in e.boxes.items():
   if len(rows)<15 or any(all(re.search(pat,docs[i]['doc'],re.I) for pat in patterns) for i in rows):continue
   words='Покажи письма о '+query+' '+suffix; d=date(2100,1,1)
   a=e.search(box,words,d); t=e.spelling.correct(words); qq=month.parse_date_scope(t,d).text
   if t!=words:changed[query]+=1
   terms={_stem(word) for word in _CYRILLIC.findall(qq.casefold().replace("ё","е")) if len(word)>=3 and word not in _STOP and word not in _BOILERPLATE and _stem(word) not in _REQUEST_WORDS}
   for did,sc in a:
    r=e.by_id[did]; raw=docs[r]['doc'];body=_BODY.split(raw,maxsplit=1)[-1]; ss=e.fields.addressed_subjects.get(r,''); present={_stem(z) for z in _CYRILLIC.findall(raw.casefold().replace('ё','е'))}&terms
    reasons[(r in e.bilingual.content, r in e.fields.addressed_subjects, bool(ss),min(5,len(_ENGLISH.findall(body[:700]))),bool(_SELECT.search(qq)),bool(present), tuple(sorted(present)))]+=1
print('reasons',reasons,'changes',changed)
