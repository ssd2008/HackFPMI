import hashlib, json, os, pathlib, subprocess, sys, tarfile, io
repo=pathlib.Path.cwd();priv=repo/'.cache/ood-mail';ext=pathlib.Path('/Users/kirill/Downloads/HackFPMI-mail-OOD/suite')
revisions={s:subprocess.check_output(['git','rev-parse',s],text=True).strip() for s in ('b37c309','5fe3f5d')}
os.umask(0o077)
for port,ds,rev in ((18776,'personal_dev','b37c309'),(18773,'vk_dev','5fe3f5d'),(18777,'personal_dev','5fe3f5d')):
 work=priv/f'server-{port}'
 work.mkdir(exist_ok=True)
 b=subprocess.check_output(['git','archive',revisions[rev],'mailsearch','static'])
 with tarfile.open(fileobj=io.BytesIO(b)) as tar:
  tar.extractall(work,filter='data')
 for name,target in (('data',ext/ds),('.cache',repo/'.cache')):
  p=work/name
  if not p.exists(): p.symlink_to(target,target_is_directory=True)
 assert (work/'data/documents.jsonl').resolve()==(ext/ds/'documents.jsonl').resolve()
 log=(priv/f'server-{port}.log').open('ab',buffering=0)
 opt=['--data',str(ext/ds)] if rev=='b37c309' else []
 cmd=[str(repo/'.venv/bin/python'),'-m','mailsearch','--port',str(port),*opt]
 env={**os.environ,'PYTHONPATH':str(work),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
 p=subprocess.Popen(cmd,cwd=work,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 meta={'revision':revisions[rev],'port':port,'source_archive_sha256':hashlib.sha256(b).hexdigest(),'corpus_sha256':hashlib.sha256((work/'data/documents.jsonl').read_bytes()).hexdigest(),'pid':p.pid}
 (priv/f'server-{port}.json').write_text(json.dumps(meta))
 print(ds,port,p.pid,revisions[rev][:8],meta['corpus_sha256'][:10])
