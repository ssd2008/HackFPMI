from pathlib import Path
import json,os,hashlib,re
root=Path(__file__).resolve().parents[3];run=root/'.cache/round4/answer/combo-61bae2d95ac6';results=root/'results';versions={'before':'a211900726b41a35912a0512ceb329f083ae6476','after':'61bae2d95ac6bf32a9bd59dbc1056173439bfae3'};load=lambda file:json.loads(file.read_bytes());pressure=load(run/'load-eight.json');assert pressure['versions']==versions

def publish(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(temp,path)

proof={}
for side,folder,ports in [('before','full',(18810,18811)),('after','final',(18820,18821))]:
 proof[side]={}
 for source,port in zip(['vk','personal'],ports):
  note=load(root/f'.cache/round4/{folder}/server-{port}.json');assert note['revision']==versions[side];proof[side][source]={k:note[k] for k in ['revision','port','source_archive_sha256','corpus_sha256']}
for split,key,filename in [('dev','full_open','ood_round4_rag.json'),('holdout','full_holdout','ood_round4_rag_holdout.json')]:
 r=load(run/(split+'.json'));assert r['versions']==versions and r['status']=='complete';r['split']=split;r['deployment_checks']=proof;r['method']['interpretation']='This is a repeated evaluation following diagnosis and development. The original frozen fixture fields and source IDs are unchanged. Results describe the registered facts in these mailboxes.'
 cache=root/'.cache/ood-mail/combo-61bae2d95ac6'
 r['serial_citation_audit']={}
 for source,cohort in r['sources'].items():
  r['serial_citation_audit'][source]={}
  for side in versions:
   files=[p for p in (cache/split/source/side).glob('*.json') if p.name!='input.json']; rows=[load(p) for p in files]
   assert len(rows)==cohort[key][side]['all']['total']
   uncited=sum(bool(line.strip()) and not re.search(r"\[\d+\]",line) for row in rows if row['mode']!='no_evidence' for line in row['answer'].splitlines())
   unused=sum(set(range(1,len(row['opened_source_ids'])+1)).difference(int(n) for n in re.findall(r"\[(\d+)\]",row['answer'])).__len__() for row in rows)
   r['serial_citation_audit'][source][side]={'uncited_nonempty_answer_lines_outside_refusals':uncited,'opened_sources_not_referenced_in_answer':unused}
 r['eight_client_recheck']={'versions':versions,'clients':pressure['clients'],'seed':pressure['seed'],'method':pressure['method'],'sources':pressure['groups'][split]}
 for source,rows in pressure['groups'][split].items():
  for side,m in rows.items(): assert m['all']['total']==r['sources'][source][key][side]['all']['total']
 extra=pressure['groups']['extra_repeat']['personal'];
 if extra['after']['fixture_split']==split:r['eight_client_recheck']['additional_personal_repeat']=extra
 publish(results/filename,r)
prior={'versions':versions,'deployment_checks':{side:pr['vk'] for side,pr in proof.items()},'protocol':'Each case was asked unchanged on both localhost /api/answer deployments, one request at a time. The registered fact patterns must appear in a line attributed to an allowed source. All cited messages were opened and their bodies, mailbox and snapshot checked. Explicit abstentions use the no_evidence response. These checks assess registered details rather than every generated statement. The four preexisting sets were revisited during diagnosis.','fixtures':{}}
for name in ['rag_facts','validation_pilot/rag_facts','rag_unseen','rag_confirmation']:
 payload={}
 for side,suffix in [('before','_before'),('after','')]:
  file=run/(name.replace('/','_')+suffix+'.json');s=load(file);actual=hashlib.sha256((root/'tests/fixtures'/(name+'.json')).read_bytes()).hexdigest();assert actual==s['fixture_sha256'];assert s['server_corpus_check']=='sha256';payload[side]=s['summary'];payload['fixture_sha256']=actual;payload['corpus_sha256']=s['corpus_sha256']
 prior['fixtures'][name]=payload
publish(results/'ood_round4_prior_rag.json',prior)
print('Written three verified aggregate reports.')
