"""Local coordinator for paired fixtures and small regressions on a pinned deployment."""
import argparse, json, hashlib, os, shutil, subprocess, sys
from pathlib import Path

base=Path(__file__).resolve().parents[3]
wt=base/'.cache/round4-answer'
sys.path.insert(0,str(wt/'tools'))
from check_mail_ood_rag import load_inputs
from check_mail_ood import archive_evidence
suite=Path('/Users/kirill/Downloads/HackFPMI-mail-OOD')
p=argparse.ArgumentParser();p.add_argument('revision');p.add_argument('vk_port',type=int);p.add_argument('personal_port',type=int);p.add_argument('serverdir',type=Path);p.add_argument('--before-holdout',type=Path);p.add_argument('--small-before',action='store_true');p.add_argument('--only',choices=['small','dev','holdout','all'],default='all');a=p.parse_args();base_sha='a211900726b41a35912a0512ceb329f083ae6476';rev=subprocess.check_output(['git','rev-parse',a.revision+'^{commit}'],cwd=base,text=True).strip();out=base/'.cache/round4/answer'/('combo-'+rev[:12]);private=base/'.cache/ood-mail'/('combo-'+rev[:12]);out.mkdir(exist_ok=True);private.mkdir(mode=0o700,parents=True,exist_ok=True);os.umask(0o077)
urls={('vk','before'):'http://127.0.0.1:18810',('personal','before'):'http://127.0.0.1:18811',('vk','after'):f'http://127.0.0.1:{a.vk_port}',('personal','after'):f'http://127.0.0.1:{a.personal_port}'}
for s in ('vk','personal'):
 ds=load_inputs(suite,s)
 for v,dir_,sha in [('before',base/'.cache/round4/full',base_sha),('after',a.serverdir,rev)]:
  assert archive_evidence(dir_, urls[s,v], sha, ds['hashes']['documents'])

def baseline_copy(prev,now,split):
 if not prev: return
 receipt={'baseline_revision':base_sha,'sources':{}}
 for source in ('vk','personal'):
  data=load_inputs(suite,source,split);old=prev/source/'before';new=now/source/'before';new.mkdir(mode=0o700,parents=True,exist_ok=True)
  stamp=json.loads((old/'input.json').read_bytes());assert stamp=={'version':base_sha,**data['hashes'],'server_corpus_check':'sha256'}
  files=sorted(old.glob('*.json')); answers=[f for f in files if f.name != 'input.json']; assert len(answers)==len(data['cases']);
  for f in files:
   dst=new/f.name
   if dst.exists():assert dst.read_bytes()==f.read_bytes()
   else:shutil.copy2(f,dst)
  receipt['sources'][source]={'reused_frozen_response_count':len(answers),'sha256_of_ordered_response_hashes':hashlib.sha256(b''.join(hashlib.sha256(f.read_bytes()).digest() for f in answers)).hexdigest()}
 (now/'baseline-reuse.json').write_text(json.dumps(receipt)+'\n')

def paired(split):
 here=private/split;here.mkdir(mode=0o700,exist_ok=True);baseline_copy(base/'.cache/round4/full/rag' if split=='dev' else a.before_holdout,here,split)
 with (out/(split+'.log')).open('w') as log:
  cmd=[sys.executable,str(wt/'tools/check_mail_ood_rag.py'),'--pool',str(suite),'--split',split,'--baseline-revision',base_sha,'--candidate-revision',rev,'--cache',str(here),'--output',str(out/(split+'.json'))]
  for (s,v),u in urls.items():cmd.extend([f'--{s}-{v}',u]);
  subprocess.run(cmd,stdout=log,check=True)
 r=json.loads((out/(split+'.json')).read_bytes());print(split,r['status'],{k:{v:row['full_open' if split=='dev' else 'full_holdout'][v]['all']['counts_by_outcome'] for v in ['before','after']} for k,row in r['sources'].items()},flush=True)

def small():
 for name in ['rag_facts','validation_pilot/rag_facts','rag_unseen','rag_confirmation']:
  for side in (('before','after') if a.small_before else ('after',)):
   dest=out/(name.replace('/','_')+('_before' if side=='before' else '')+'.json')
   with dest.with_suffix('.log').open('w') as log:
    subprocess.run([sys.executable,str(wt/'tools/check_rag_facts.py'),'--fixture',str(wt/'tests/fixtures'/(name+'.json')),'--documents',str(base/'data/documents.jsonl'),'--url',urls['vk',side],'--output',str(dest)],check=True,stdout=log)
   res=json.loads(dest.read_bytes())['summary'];print(name,side,res['all_facts_with_expected_citation'],res['answerable'],res['explicit_no_evidence'],res['insufficient'],flush=True)
for phase in (['holdout','small','dev'] if a.only=='all' else [a.only]):
 if phase=='small':small()
 else:paired(phase)
