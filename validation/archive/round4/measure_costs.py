import cProfile,pstats,json,sys,hashlib
from pathlib import Path
from datetime import date
root=Path.cwd();sys.path.insert(0,str(root))
from mailsearch.retrieval import SearchEngine
pool=Path.home()/'Downloads/HackFPMI-mail-OOD'
def read(f):return [json.loads(l) for l in f.open() if l.strip()]
for n in ('vk_dev','personal_dev'):
 ds=pool/'suite'/n;docs=read(ds/'documents.jsonl');e=SearchEngine(docs,root/'.cache/mailsearch')
 qs=read(ds/'queries_dev.jsonl')+read(ds/'queries_empty.jsonl')+read(ds/'queries_variants.jsonl')
 qs=sorted(qs,key=lambda q:hashlib.sha256(q['query_id'].encode()).digest())[:300]
 work=[(n,qs),('public',read(root/'data/queries_dev.jsonl'))] if n=='vk_dev' else [(n,qs)]
 for name,queries in work:
  profile=cProfile.Profile();profile.enable()
  for q in queries: e.search(q['synthetic_user'],q['query'],date.fromisoformat(q['query_date']),10)
  profile.disable();print('Workload',name,flush=True)
  pstats.Stats(profile,stream=sys.stdout).strip_dirs().sort_stats('cumtime').print_stats(25)
