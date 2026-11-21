from pathlib import Path
import subprocess,json,hashlib,tarfile
p=Path('.cache/round4/field')
for name in ('rus-eng','eng-rus'):
 a=next(d for d in json.loads((p/'freedict-index.json').read_text()) if d.get('name')==name);src=next(v for v in a['releases'] if v['platform']=='src');print(name,a['headwords'],src['size'])
 target=p/(name+'.tar.xz')
 subprocess.run(['curl','-fsSL','--retry','2','--max-time','20',src['URL'],'-o',str(target)],check=True)
 assert hashlib.sha512(target.read_bytes()).hexdigest()==src['checksum']
 with tarfile.open(target) as tx:
  files=[x.name for x in tx.getmembers() if x.isfile()]
  print(files[:5])
  tei=next(n for n in files if n.endswith('.tei'))
  (p/(name+'.tei')).write_bytes(tx.extractfile(tei).read())
