from pathlib import Path
import sys,json,hashlib,os
from concurrent.futures import ThreadPoolExecutor
root=Path(__file__).resolve().parents[3];wt=root/'.cache/round4-answer';sys.path.insert(0,str(wt/'tools'));from check_mail_ood_rag import load_inputs;from check_rag_facts import get
private=root/'.cache/ood-mail/combo-9a720b79d78a';pool=Path('/Users/kirill/Downloads/HackFPMI-mail-OOD');os.umask(0o077)
for source,port in [('vk',18817),('personal',18818)]:
 for split in ('dev','holdout'):
  data=load_inputs(pool,source,split);out=private/split/source/'top5.json';
  def call(r):
   cid,q=r;ans=get(f'http://127.0.0.1:{port}','/api/search',{'mailbox':q['mailbox'],'date':q['date'],'query':q['query'],'limit':5});return cid,[v['id'] for v in ans['items']]
  with ThreadPoolExecutor(max_workers=4) as ex: saved=dict(ex.map(call,data['cases'].items()))
  out.write_text(json.dumps(saved)+'\n');print(source,split,len(saved),flush=True)
