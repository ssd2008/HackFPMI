import pathlib,os,sys,subprocess
root=pathlib.Path.cwd();work=root/'.cache/round4-mail-rank';p=pathlib.Path('/Users/kirill/Downloads/HackFPMI-mail-OOD')
runs=root/'.cache/round4/field'/sys.argv[1];os.umask(0o077)
for name,port in (('vk_dev',18801),('personal_dev',18802)):
 out=runs/name;out.mkdir(exist_ok=True,parents=True)
 with (out/'http.log').open('wb') as log:
  proc=subprocess.Popen([str(root/'.venv/bin/python'),'-u',str(p/'method/run_pool.py'),'--project',str(work),'--suite',str(p/'suite'/name),'--url',f'http://127.0.0.1:{port}','--output',str(out),'--workers','2','--variants'],stdout=log,stderr=log,stdin=subprocess.DEVNULL,cwd=work,start_new_session=True)
  print(name,proc.pid)
