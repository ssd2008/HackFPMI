import sys, time, random, os, cProfile, pstats, statistics, json, dataclasses, pathlib, re
from datetime import date
from collections import Counter,defaultdict
root=pathlib.Path.cwd(); scratch=root/'.cache/round4/field/quote-speed'
pool=pathlib.Path.home()/'Downloads/HackFPMI-mail-OOD';suite=pool/'suite/personal_dev';label=sys.argv[1]
work=root/('.cache/round4/final/server-18821' if label=='before' else '.cache/round4-quote-speed')
sys.path.insert(0,str(work));import mailsearch.retrieval as m
read=lambda p:[json.loads(i) for i in p.open()]
docs=read(suite/'documents.jsonl');engine=m.SearchEngine(docs,root/'.cache/mailsearch')
pids={r['query_id'] for r in read(pool/'panels/personal_dev.jsonl')}
qs={q['query_id']:q for name in ('dev','empty') for q in read(suite/f'queries_{name}.jsonl') if q['query_id'] in pids}
fn=lambda q:engine.search(q['synthetic_user'],q['query'],date.fromisoformat(q['query_date']),10)
ref=json.loads((root/'.cache/round4/final/panels/personal_dev/after/rankings.json').read_bytes())['queries']
dur=defaultdict(list);diff=set()
for j in range(4):
 order=list(qs);random.Random(2026+j).shuffle(order)
 for qid in order:
  t=time.perf_counter_ns();out=fn(qs[qid]);ms=(time.perf_counter_ns()-t)/1e6
  if j>0:dur[qid].append(ms)
  if [i for i,_ in out]!=ref[qid]['ranked_ids']:diff.add(qid)
top=sorted(dur,key=lambda i:-statistics.median(dur[i]))[:3]
result={'duration_summary':{str(idx):round(statistics.median(dur[q]),3) for idx,q in enumerate(top)},'n':len(qs),'diff':len(diff),'profile':[]}
for qid in top:
 pr=cProfile.Profile();pr.enable()
 for j in range(4):fn(qs[qid])
 pr.disable();st=pstats.Stats(pr)
 arr=[(v[3]/4*1000, ' '.join((pathlib.Path(k[0]).name,str(k[1]),k[2]))[:105],round(v[2]/4*1000,3),int(v[1]/4)) for k,v in st.stats.items()]
 result['profile'].append(sorted(arr,reverse=True)[:12])
(scratch/f'profile-{label}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
