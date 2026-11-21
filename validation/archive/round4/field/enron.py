import sys, time, json, pathlib, os, statistics, random
from datetime import date
root=pathlib.Path.cwd(); folder=root/'.cache/round4/field'
work=root if sys.argv[1]=='base' else root/'.cache/round4-mail-rank'
sys.path.insert(0,str(work))
from mailsearch.retrieval import SearchEngine
if len(sys.argv)>2 and sys.argv[2]=="main-dates":
 import importlib.util,mailsearch.retrieval
 spec=importlib.util.spec_from_file_location("round_dates",root/"mailsearch/dates.py");month=importlib.util.module_from_spec(spec);sys.modules[spec.name]=month;spec.loader.exec_module(month);mailsearch.retrieval.parse_date_scope=month.parse_date_scope
s=pathlib.Path('/Users/kirill/Downloads/HackFPMI-mail-OOD/external/benchmarks/enron-mail')
load=lambda p:[json.loads(l) for l in p.open()]
engine=SearchEngine(load(s/'documents.jsonl'),root/'.cache/ood-transfer-small-candidate/vectors/enron-mail')
out={}; times=[]
for ch in ('dev','empty','variants'):
 qs=load(s/f'queries_{ch}.jsonl');random.Random(2026).shuffle(qs);out[ch]={}
 for q in qs:
  a=time.perf_counter()
  ranks=engine.search(q['synthetic_user'],q['query'],date.fromisoformat(q['query_date']),10)
  times.append((time.perf_counter()-a)*1000)
  out[ch][q['query_id']]=[did for did,score in ranks]
(folder/f'enron-{sys.argv[1]}.json').write_text(json.dumps({'ranked_ids':out,'latency':statistics.quantiles(times,n=100,method='inclusive')[98]}))
print('done',sys.argv[1],len(times),flush=True)
