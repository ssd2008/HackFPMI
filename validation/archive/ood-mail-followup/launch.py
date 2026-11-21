"""Start loopback servers from one frozen revision, with the corpus verified on each port."""
from pathlib import Path
import hashlib, io, json, os, socket, subprocess, sys, tarfile
repo=Path.cwd(); root=repo/'.cache/ood-mail-followup'
pool=Path.home()/'Downloads/HackFPMI-mail-OOD/suite'
rev=subprocess.check_output(['git','rev-parse','--verify',sys.argv[1]+'^{commit}'],text=True).strip()
os.umask(0o077)
package=subprocess.check_output(['git','archive',rev,'mailsearch','static'])
servers = [('vk_dev',18782)] if '--baseline-vk' in sys.argv[2:] else [('vk_dev',18780),('personal_dev',18781)]
for name,port in servers:
 with socket.socket() as guard:
  guard.bind(('127.0.0.1',port))
 work=root/f'server-{port}';work.mkdir(exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(package)) as archive: archive.extractall(work,filter='data')
 for p,target in [(work/'data',pool/name),(work/'.cache',repo/'.cache')]:
  if not p.exists(): p.symlink_to(target,target_is_directory=True)
 expected=json.loads((pool/name/'manifest.json').read_bytes())['sha256']['documents.jsonl']
 with (work/'data/documents.jsonl').open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==expected
 env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','PYTHONPATH':str(work)}
 with (root/f'server-{port}.log').open('ab',buffering=0) as log:
  proc=subprocess.Popen([str(repo/'.venv/bin/python'),'-m','mailsearch','--port',str(port),'--data',str(pool/name)],cwd=work,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 note={'revision':rev,'port':port,'source_archive_sha256':hashlib.sha256(package).hexdigest(),'corpus_sha256':expected,'pid':proc.pid}
 (root/f'server-{port}.json').write_text(json.dumps(note)+'\n')
 print(name,port,proc.pid)
