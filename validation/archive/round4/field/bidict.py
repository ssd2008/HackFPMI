import sys,re,collections,unicodedata,gzip,json
from pathlib import Path
from xml.etree.ElementTree import iterparse
p=Path('.cache/round4/field');sys.path.insert(0,str(Path.cwd()/'.cache/round4-mail-rank'))
from mailsearch.retrieval import _stem
def norm(w):return unicodedata.normalize('NFC',w.casefold().replace('ё','е').replace('\u0301','')).strip()
directions=[]
for name in ('rus-eng','eng-rus'):
 pairs=collections.defaultdict(dict)
 for event,e in iterparse(p/(name+'.tei'),events=['end']):
  if not e.tag.endswith('}entry'):continue
  head=norm(e.findtext('./{*}form/{*}orth','')); words=[norm(v.text) for v in e.findall('.//{*}cit[@type="trans"]/{*}quote') if v.text]
  for trans in words:
   ru,en=(head,trans) if name=='rus-eng' else (trans,head);en=en.removeprefix('to ')
   if re.fullmatch('[а-я]{3,}',ru) and re.fullmatch(r'[a-z]+(?:[ -][a-z]+){0,2}',en):
    ru=_stem(ru);a,b=(ru,en) if name=='rus-eng' else (en,ru)
    group=pairs[a];group.setdefault(b,len(group))
  e.clear()
 directions.append(pairs)
f,r=directions
for mode in ('top','mutual','conservative'):
 o=collections.defaultdict(list)
 for ru,glosses in f.items():
  for en,i in glosses.items():
   inv=r.get(en,{})
   if mode=='mutual' and ru not in inv:continue
   if mode=='top' and i>=3:continue
   if mode=='conservative' and not (ru in inv and (inv[ru]<4 or len(inv)<=6) or (i<2 and en not in r)): continue
   o[ru].append(en)
 with gzip.GzipFile(str(p/f'lex-{mode}.tsv.gz'),'wb',mtime=0) as dest:
  dest.write(''.join(k+'\t'+'|'.join(es)+'\n' for k,es in sorted(o.items())).encode())
 print(mode,len(o),[o.get(_stem(w),[]) for w in ['шахматы','телескоп','счет']])
