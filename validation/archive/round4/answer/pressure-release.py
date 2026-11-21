"""Run both complete fixed RAG splits on one frozen deployment pair with eight local workers."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from collections import Counter
import argparse,hashlib,json,os,random,sys
from math import ceil
root=Path(__file__).resolve().parents[3];wt=root/'.cache/round4-answer';sys.path.insert(0,str(wt/'tools'));from check_mail_ood_rag import load_inputs,grade,aggregate,server_check,reply_signature
from check_rag_facts import check
p=argparse.ArgumentParser();p.add_argument('revision');p.add_argument('vk_port',type=int);p.add_argument('personal_port',type=int);p.add_argument('--reuse-before',required=True);a=p.parse_args();base='a211900726b41a35912a0512ceb329f083ae6476';pool=Path('/Users/kirill/Downloads/HackFPMI-mail-OOD');folder=root/'.cache/round4/answer'/('combo-'+a.revision[:12]);raw=root/'.cache/ood-mail'/('combo-'+a.revision[:12]);urls={'vk':{'before':'http://127.0.0.1:18810','after':f'http://127.0.0.1:{a.vk_port}'},'personal':{'before':'http://127.0.0.1:18811','after':f'http://127.0.0.1:{a.personal_port}'}};os.umask(0o077);changes={};size=8;seed=20261121
prev_run=root/'.cache/round4/answer'/('combo-'+a.reuse_before[:12]); prev_raw=root/'.cache/ood-mail'/('combo-'+a.reuse_before[:12]); prev_file=prev_run/'load-eight.json'; prev_pressure=json.loads(prev_file.read_bytes()); assert prev_pressure['versions']=={'before':base,'after':a.reuse_before} and prev_pressure['clients']==size and prev_pressure['seed']==seed
reused={'revision':base,'previous_paired_candidate':a.reuse_before,'previous_full_aggregate_sha256':hashlib.sha256(prev_file.read_bytes()).hexdigest(),'per_corpus':{}}
for source in ('personal','vk'):
 data={s:load_inputs(pool,source,s) for s in ('dev','holdout')};assert data['dev']['hashes']['documents']==data['holdout']['hashes']['documents'];all_cases=[(s,cid) for s in ('dev','holdout') for cid in data[s]['cases']];schedule=all_cases.copy();random.Random(seed).shuffle(schedule);original_size=len(schedule)
 # Seven personal questions across the two splits. An eighth, preassigned repeat fills a wave and is counted separately.
 while len(schedule)<size:schedule.append(schedule[len(schedule)-original_size]);
 for side in (('after','before') if source=='personal' else ('before','after')):
  out=raw/'load-eight'/source/side;out.mkdir(mode=0o700,parents=True,exist_ok=True);full=[];barrier=Barrier(size);url=urls[source][side]
  if side=='after':assert server_check(url,data['dev'])=='sha256'
  else:
   old=prev_raw/'load-eight'/source/'before'; ordered=[old/(str(i)+'.json') for i in range(len(schedule))]; assert len(list(old.glob('*.json')))==len(ordered); reused['per_corpus'][source]={'replayed_records':len(ordered),'sha256_of_response_digests_in_original_dispatch_order':hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in ordered)).hexdigest()}
  def run(item):
   ix,(s,cid)=item;d=data[s];c=d['cases'][cid];
   if side=='after':
    if ix<size:barrier.wait(timeout=20)
    result=check(url,c,d['docs']);dest=out/(str(ix)+'.json');dest.write_text(json.dumps(result,ensure_ascii=False)+'\n')
   else:
    result=json.loads(ordered[ix].read_bytes());assert result['id']==cid
   grade_row=grade(result,c,d['metadata'][cid],d['docs']);pr=raw/s/source/side/(hashlib.sha256(cid.encode()).hexdigest()+'.json');serial=json.loads(pr.read_bytes());sv=grade(serial,c,d['metadata'][cid],d['docs']);return s,ix>=original_size,grade_row,{'identical_text_sources_mode':reply_signature(result)==reply_signature(serial),'same_quality_outcome':grade_row['class']==sv['class'],'same_ordered_sources':result['opened_source_ids']==serial['opened_source_ids'],'same_mode':result['mode']==serial['mode']}
  with ThreadPoolExecutor(max_workers=size) as ex:
   for i,record in enumerate(ex.map(run,enumerate(schedule)),1):
    full.append(record)
    if not i%100: print(source,side,i,len(schedule),flush=True)
  for spl in ['dev','holdout','extra_repeat']:
   sample=[(v,match) for s,extra,v,match in full if (spl=='extra_repeat' and extra or spl==s and not extra)]
   if sample:
    summary=aggregate([v for v,_ in sample]); compare={k:sum(m[k] for _,m in sample) for k in sample[0][1]}; times=sorted(v['answer_ms'] for v,_ in sample)
    changes.setdefault(spl,{}).setdefault(source,{})[side]={'all':summary,'against_serial':compare,'answer_http_ms':{f'p{int(p*100)}':times[ceil(p*len(times))-1] for p in [.5,.95,.99]}}
  if len(schedule)>original_size: changes['extra_repeat'][source][side]['fixture_split']=schedule[original_size][0]
  print(source,side,{s:(d[source][side]['all']['full_grounded_answers'],d[source][side]['all']['justified_abstentions']) for s,d in changes.items() if source in d and side in d[source]},flush=True)
for split,sources in changes.items():
 for source,d in sources.items(): assert d['before']==prev_pressure['groups'][split][source]['before']
report={'baseline_reuse':reused, 'versions':{'before':base,'after':a.revision},'clients':size,'seed':seed,'method':'Eight concurrent local /api/answer workers per corpus and revision. Each of the 388 registered questions is asked once, shuffled across splits within each corpus. For seven personal questions a single separately reported duplicate fills the eight-request wave. Citation links are opened in the original scope. Eight initial requests are dispatched together; the closed-loop pool processes the remaining VK questions. Concurrent answer timings are distinct from search latency.','groups':changes}
(folder/'load-eight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
