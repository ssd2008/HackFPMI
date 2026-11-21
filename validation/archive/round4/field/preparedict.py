import os,pickle,json,gzip,re,collections,unicodedata,sys
from pathlib import Path
from xml.etree.ElementTree import iterparse
root=Path.cwd();p=root/'.cache/round4/field'
sys.path.insert(0,str(root/'.cache/round4-mail-rank'))
from mailsearch.retrieval import _stem
out=collections.defaultdict(list)
def clean(w): return unicodedata.normalize('NFC',w.casefold().replace('ё','е').replace('\u0301','')).strip()
for name in ('rus-eng','eng-rus'):
 for _,elem in iterparse(p/f'{name}.tei',events=['end']):
  if not elem.tag.endswith('}entry'):continue
  h=elem.findtext('./{*}form/{*}orth','');translations=[x.text for x in elem.findall('.//{*}cit[@type="trans"]/{*}quote') if x.text]
  for t in translations:
   ru,en=(h,t) if name=='rus-eng' else (t,h)
   ru=clean(ru);en=clean(en)
   if not re.fullmatch('[а-я]{3,}',ru) or not re.fullmatch(r'(?:to )?[a-z]+(?:[ -][a-z]+){0,2}',en):continue
   en=en.removeprefix('to ')
   k=_stem(ru)
   if en not in out[k]:out[k].append(en)
  elem.clear()
 with gzip.GzipFile(str(p/f'lex-{name}.tsv.gz'),'wb',mtime=0) as f:
  f.write(''.join(k+'\t'+'|'.join(v)+'\n' for k,v in sorted(out.items())).encode())
 print(name,len(out),sum(map(len,out.values())),flush=True)
for t in ['выставленных','счетах','отчётах','расходам','повесткой','трубопроводах','сбой','отключениях','хамелеоны','кактусы','шахматам','ремонте','дирижабля']:
 print(t,_stem(t.replace('ё','е')),out.get(_stem(t.replace('ё','е')),[])[:8])
