import os,subprocess,pathlib,sys
root=pathlib.Path.cwd();work=root/'.cache/round4-mail-rank';logdir=root/'.cache/round4/field'
suite=pathlib.Path('/Users/kirill/Downloads/HackFPMI-mail-OOD/suite')
for port,name in [(18801,'vk_dev'),(18802,'personal_dev')]:
 cmd=[str(root/'.venv/bin/python'),'-c','import sys,uvicorn; from pathlib import Path; import mailsearch.app as s; s.CACHE=Path(sys.argv[1]); s.app.state.data_dir=Path(sys.argv[2]); uvicorn.run(s.app,host="127.0.0.1",port=int(sys.argv[3]),access_log=False)',str(root/'.cache/mailsearch'),str(suite/name),str(port)]
 with (logdir/f'{port}.log').open('ab') as log:
  proc=subprocess.Popen(cmd,stdout=log,stderr=log,stdin=subprocess.DEVNULL,cwd=work,start_new_session=True)
  (logdir/f'{port}.pid').write_text(str(proc.pid));print(port,proc.pid)
