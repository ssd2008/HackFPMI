import sys,time,random,pathlib,json,statistics,collections
from concurrent.futures import ThreadPoolExecutor
from datetime import date
root=pathlib.Path.cwd(); scratch=root/'.cache/round4/field/quote-speed';sys.path.insert(0,str(root))
from benchmark import fetch,percentiles
pool=pathlib.Path.home()/'Downloads/HackFPMI-mail-OOD'
read=lambda p:[json.loads(i) for i in p.open()]
pids={r['query_id'] for r in read(pool/'panels/personal_dev.jsonl')}
qs={q['query_id']:q for ch in ('dev','empty') for q in read(pool/'suite/personal_dev'/f'queries_{ch}.jsonl') if q['query_id'] in pids}
ranks=json.loads((root/'.cache/round4/final/panels/personal_dev/after/rankings.json').read_bytes())['queries']
urls={'before':'http://127.0.0.1:18821','after':'http://127.0.0.1:18862'}
clients=[int(x) for x in sys.argv[1].split(',')]; sides=sys.argv[2:]; out={}
for c in clients:
 sample={s:[] for s in sides}; mismatch={s:set() for s in sides}; bins=collections.defaultdict(list)
 for pair in range(3):
  datasets=list(qs);random.Random(20261121+c*100+pair).shuffle(datasets); datasets*= 1 if pair==0 else 4
  for tag in (sides if pair%2==0 else sides[::-1]):
   def job(qid):
    before=time.perf_counter_ns();ids,server,_=fetch(urls[tag],qs[qid],10);total=(time.perf_counter_ns()-before)/1e6
    return qid,list(ids),server,total
   with ThreadPoolExecutor(c) as executor:
    for qid,ids,ms,wall in executor.map(job,datasets):
     if ranks[qid]['ranked_ids']!=ids:mismatch[tag].add(qid)
     if pair>0:sample[tag].append((ms,wall));bins[(tag,qid)].append(ms)
 for tag in sides:
  x=sample[tag];z=sorted([statistics.median(t) for (s,_),t in bins.items() if s==tag],reverse=True)
  out[tag,c]={'server':percentiles([a for a,b in x]),'wall':percentiles([b for a,b in x]),'n':len(x),'changed_from_saved_250':len(mismatch[tag]),'slowest_med':round(z[0],3)}
 print('completed', c,{str(k):v for k,v in out.items()},flush=True)
(scratch/('http-'+'-'.join(sides)+'.json')).write_text(json.dumps({':'.join(map(str,k)):v for k,v in out.items()},indent=2))
