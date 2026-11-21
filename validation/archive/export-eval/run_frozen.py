"""Keep all nonaggregate export diagnostics under .cache."""
import importlib.util
import sys
import time
import re
import json
import math
import hashlib
import traceback
import subprocess
from pathlib import Path
folder=Path(sys.argv[1]);port=int(sys.argv[2]);label=sys.argv[3];fixture=sys.argv[4] if len(sys.argv)>4 else 'probes.json'
s=importlib.util.spec_from_file_location('checks',Path('tools/check_search_probes.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
base=f'http://127.0.0.1:{port}'
previous=m.urlopen
values=[]
def counted(req,timeout):
    resp=previous(req,timeout=timeout)
    if req.full_url==base+'/api/search':
        x=re.fullmatch(r'search;dur=([\d.]+)',resp.headers.get('Server-Timing',''))
        if x:values.append(float(x[1]))
    return resp
m.urlopen=counted
sys.argv=['check','--url',base,'--documents',str(folder/'corpus/documents.jsonl'),'--fixture',str(folder/fixture),'--output',str(folder/(label+'.json'))]
try:m.main()
except Exception:
    print('Diagnostic hit an infrastructure or fixture error. Private log saved locally.')
    (folder/(label+'-error.log')).write_text(traceback.format_exc());sys.exit(1)
j=json.loads((folder/(label+'.json')).read_text())
groups={}
for family in sorted({r['family'] for r in j['cases']}):
    rows=[r for r in j['cases'] if r['family']==family];groups[family]=m.summarize(rows,[])
values.sort();assert len(values)==len(j['cases'])
lat={f'p{q}_ms':values[math.ceil(len(values)*q/100)-1] for q in (50,95,99)}
fingerprint=hashlib.sha256()
for f in ['mailsearch/app.py','mailsearch/retrieval.py','mailsearch/dense.py','mailsearch/dates.py','mailsearch/store.py']:
    fingerprint.update(f.encode());fingerprint.update(Path(f).read_bytes())
summary={'metrics':j['metrics'],'groups':groups,'latency':lat,'distinct_sequential_requests':len(values),'tested_at_utc':j['tested_at_utc'],'fixture_sha256':j['fixture_sha256'],'code_on_disk_sha256':fingerprint.hexdigest(),'head_at_run':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
(folder/(label+'-summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
for group,g in groups.items():print(group,g['hit_at_1'],g['hit_at_5'],'of',g['positive_queries'],'empty',g['correctly_empty'],'of',g['empty_queries'])
print('Sequential Server-Timing',lat)
