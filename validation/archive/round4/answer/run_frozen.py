from pathlib import Path
import hashlib,json,subprocess,sys,time
from urllib.request import urlopen
work=Path(__file__).resolve().parent;root=work.parents[2];wt=root/'.cache/round4-answer'
sig=hashlib.sha256((wt/'mailsearch/generation.py').read_bytes()).hexdigest()
assert sig=='444a9d32f9dc44453b77381d61d2d41496d4394cc7c05424ef059e8aa550cbc5'
for port in (18803,18804):
 for idx in range(90):
  try:
   with urlopen(f'http://127.0.0.1:{port}/api/mailboxes',timeout=3) as f:json.load(f)
   break
  except OSError:time.sleep(1)
 else:raise RuntimeError('service did not become ready')
run=work/('v-'+sig[:12]);run.mkdir(exist_ok=True); py=sys.executable
subprocess.run([py,str(work/'check_round.py'),'full'],check=True)
for name in ('rag_facts','validation_pilot/rag_facts','rag_unseen','rag_confirmation'):
 with (run/(name.replace('/','_')+'.log')).open('w') as out:
  subprocess.run([py,str(wt/'tools/check_rag_facts.py'),'--url','http://127.0.0.1:18803','--fixture',str(wt/'tests/fixtures'/(name+'.json')),'--documents',str(root/'data/documents.jsonl'),'--output',str(run/(name.replace('/','_')+'.json'))],check=True,stdout=out)
 print('frozen control',name,flush=True)
subprocess.run([py,str(work/'check_round.py'),'load'],check=True)
