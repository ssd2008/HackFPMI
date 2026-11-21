from pathlib import Path
import os,subprocess,json
repo=Path.cwd();ood=Path('/Users/kirill/Downloads/HackFPMI-mail-OOD');cache=repo/'.cache/ood-mail';os.umask(0o077)
for ds,side,port in [('vk_dev','after',8766),('personal_dev','after',18776),('vk_dev','before',18773),('personal_dev','before',18777)]:
 out=cache/ds/side
 out.mkdir(parents=True,exist_ok=True)
 assert not (out/'rankings.json').exists()
 log=(out/'pool.log').open('ab',buffering=0)
 cmd=[str(repo/'.venv/bin/python'),'-u',str(ood/'method/run_pool.py'),'--project',str(repo),'--suite',str(ood/'suite'/ds),'--url',f'http://127.0.0.1:{port}','--output',str(out),'--workers','2','--variants']
 p=subprocess.Popen(cmd,cwd=repo,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 (out/'process.json').write_text(json.dumps({'pid':p.pid,'port':port}))
 print(ds,side,p.pid)
