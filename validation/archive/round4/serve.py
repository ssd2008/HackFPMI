from pathlib import Path
import hashlib,io,json,os,socket,subprocess,sys,tarfile,time
from urllib.request import urlopen
repo=Path.cwd()
if len(sys.argv)<4: raise SystemExit('serve.py RUN_DIRECTORY COMMIT SUITE=PORT ...')
base=Path(sys.argv[1]).resolve();base.mkdir(mode=0o700,exist_ok=True,parents=True);os.umask(0o077)
rev=subprocess.check_output(['git','rev-parse','--verify',sys.argv[2]+'^{commit}'],text=True).strip()
package=subprocess.check_output(['git','archive',rev,'mailsearch','static'])
pool=Path.home()/'Downloads/HackFPMI-mail-OOD'
ports=[]
for spec in sys.argv[3:]:
 name,number=spec.rsplit('=',1);port=int(number)
 ds=(pool/'suite'/name).resolve()
 if name=='public': ds=repo/'data'
 if not ds.is_dir():raise ValueError(name)
 with socket.socket() as sock:sock.bind(('127.0.0.1',port))
 dest=base/f'server-{port}';dest.mkdir(exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(package)) as archive:archive.extractall(dest,filter='data')
 for name_,target in [('data',ds),('.cache',repo/'.cache')]:
  p=dest/name_
  if not p.exists():p.symlink_to(target,target_is_directory=True)
 assert (dest/'data').resolve()==ds
 with (ds/'documents.jsonl').open('rb') as f:checksum=hashlib.file_digest(f,'sha256').hexdigest()
 assert json.loads((ds/'manifest.json').read_bytes())['sha256']['documents.jsonl']==checksum
 env={**os.environ,'PYTHONPATH':str(dest),'TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
 with (base/f'server-{port}.log').open('ab',buffering=0) as log:
  p=subprocess.Popen([str(repo/'.venv/bin/python'),'-m','mailsearch','--port',str(port),'--data',str(ds)],cwd=dest,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 note={'revision':rev,'port':port,'pid':p.pid,'source_archive_sha256':hashlib.sha256(package).hexdigest(),'corpus_sha256':checksum}
 (base/f'server-{port}.json').write_text(json.dumps(note)+'\n');ports.append((port,p,checksum))
for port,p,checksum in ports:
 for i in range(150):
  try:
   with urlopen(f'http://127.0.0.1:{port}/api/mailboxes',timeout=2) as r: assert json.load(r)['corpus_sha256']==checksum
   print(rev[:7],port,'ready',flush=True);break
  except OSError:
   if p.poll() is not None:raise RuntimeError('Server failed '+str(port))
   time.sleep(.5)
 else: raise RuntimeError('Timeout '+str(port))
