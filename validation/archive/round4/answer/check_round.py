import argparse,hashlib,json,os,random,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
ROOT=Path(__file__).resolve().parents[3]; WT=ROOT/'.cache/round4-answer'; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(WT/'tools'))
from check_mail_ood_rag import load_inputs, grade, aggregate, server_check
from check_rag_facts import check
os.umask(0o077)
pool=Path('/Users/kirill/Downloads/HackFPMI-mail-OOD')
sha=hashlib.sha256((WT/'mailsearch/generation.py').read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['develop','full','load','full-load']);p=parser.parse_args().phase
run=OUT/('v-'+sha[:12]);run.mkdir(exist_ok=True); result={}
for src,port in [('vk',18803),('personal',18804)]:
 data=load_inputs(pool,src);url=f'http://127.0.0.1:{port}';server_check(url,data);cases=data['cases'];meta=data['metadata'];docs=data['docs']
 natural=[i for i,m in meta.items() if m['family']=='reviewed_semantic']
 if p=='develop':
  ids=natural+[i for i,m in meta.items() if m['family']=='order_total'][:3]+[i for i,m in meta.items() if m['family']=='tracking_number'][:3]
 elif p=='load':
  ids=(natural+[i for i,m in meta.items() if m['family']=='order_total'][:10]+[i for i,m in meta.items() if m['family']=='tracking_number'][:8])*2
  if src=='personal':ids=natural*8
  random.Random(20261121).shuffle(ids)
 else:
  ids=list(cases)
  if p=='full-load':random.Random(20261122).shuffle(ids)
 def issue(x):
  ix,cid=x;folder=run/(src+'-serial' if p in {'develop','full'} else src+'-'+p);folder.mkdir(exist_ok=True)
  filename=hashlib.sha256(cid.encode()).hexdigest() if p in {'develop','full'} else str(ix)
  tgt=folder/(filename+'.json')
  if p in {'develop','full'} and tgt.exists():row=json.loads(tgt.read_text())
  else:
   row=check(url,cases[cid],docs);tgt.write_text(json.dumps(row,ensure_ascii=False)+'\n')
  verdict=grade(row,cases[cid],meta[cid],docs)
  if p=='develop':print(src,meta[cid]['family'],verdict['class'],row['mode'],row['facts_cited'],row['facts_expected'],row['citations_to_expected'],row['citations'],flush=True)
  return verdict
 with ThreadPoolExecutor(max_workers=8 if p in {'load','full-load'} else 1) as executor:
  rows=[]
  for ix,row in enumerate(executor.map(issue, enumerate(ids)),1):
   rows.append(row)
   if ix%50==0:print(src,ix,len(ids),flush=True)
 result[src]={'all':aggregate(rows),'by_family':{f:aggregate([r for r in rows if r['family']==f]) for f in sorted(set(r['family'] for r in rows))}}
print({s:(v['all']['counts_by_outcome'],v['all']['modes']) for s,v in result.items()},flush=True)
(run/(p+'.json')).write_text(json.dumps({'generation_sha256':sha,'phase':p,'concurrency':8 if p in {'load','full-load'} else 1,'sources':result},ensure_ascii=False,indent=2)+'\n')
