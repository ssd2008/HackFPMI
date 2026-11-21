import json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
from hashlib import sha256
cache=Path('.cache/ood-mail-followup')
revs={'before':'b37c309ff3d9e83f80a726325a3b9b0ee56c33c1','after':'b5252c144fc709168c96f8957d97c1d9ee5353af'}; ports={'before':18782,'after':18780}
rec={'revisions':revs,'measured_at':datetime.now(timezone.utc).isoformat(),'protocol':'Two warmed localhost servers on the identical official corpus. Five repetitions at each concurrency in each batch, two batches in reverse engine order. Server-Timing and loopback HTTP in milliseconds. All 99 original rankings verified across all batches and versions.','batches':[]}
canonical=None
for clients in (1,8):
 for order in (('before','after'),('after','before')):
  for side in order:
   out=cache/('public-batch-'+str(len(rec['batches']))+'.json')
   cmd=[sys.executable,'benchmark.py','--data','data','--url','http://127.0.0.1:'+str(ports[side]),'--k','1','5','10','--repeats','5','--concurrency',str(clients),'--output',str(out)]
   with (cache/'public-load.log').open('a') as log: subprocess.run(cmd,stdout=log,stderr=log,check=True)
   d=json.loads(out.read_bytes()); key='queries'; ranks={qid:v['ranked_ids'] for qid,v in d[key].items()}
   if canonical is None: canonical=ranks
   assert len(ranks)==99 and canonical==ranks and d['timing']['ranking_changes_on_repeats']==0
   if 'quality' not in rec:
    rec['quality']=d['quality_first_pass'];rec['recall_upper_bound']=d['recall_upper_bound'];rec['dataset']=d['dataset']
   assert rec['quality']==d['quality_first_pass']
   rec['batches'].append({'engine':side,'clients':clients,**{k:d['timing'][k] for k in ['repeats','search_calls','server_ms','loopback_http_ms','ranking_changes_on_repeats','platform','python']}})
   print(side,clients,d['timing']['server_ms']['p99'],flush=True)
rec['identical_top_10_queries']=len(canonical)
Path('results/ood_public_followup.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n')
