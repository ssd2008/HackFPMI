import collections,hashlib,json,math,sys
from pathlib import Path
work=Path(__file__).resolve().parent;root=work.parents[2];w=root/'.cache/round4-answer'
sha=hashlib.sha256((w/'mailsearch/generation.py').read_bytes()).hexdigest();run=work/('v-'+sha[:12])
load=lambda f:json.loads(f.read_bytes())
summary={'baseline_tree':'a211900726b41a35912a0512ceb329f083ae6476','generation_sha256':sha,'development':'Open VK/personal fixture only and invented synthetic counterexamples. The previous unseen/confirmation answers were held out of design.',
         'data_scope':'All personal texts, identifiers and responses are local. This report publishes only aggregate counts.',
         'method':'Real local /api/answer on the same base retriever and corpus. A complete result includes all requested facts, a relevant link and no conflicting or extraneous cited answer. Each returned link was opened and compared with the source body in the same mailbox and date.',
         'phases':{},'frozen_regressions':{}}
for phase in ('full','load','full-load'):
 data=load(run/(phase+'.json'));out={}
 for source,record in data['sources'].items():
  sub=run/(source+'-'+('serial' if phase=='full' else phase))
  rows=[load(p) for p in sub.glob('*.json')]
  metric=record['all'];assert len(rows)==metric['total']
  times=sorted(r['answer_http_ms'] for r in rows)
  original_by_id={r['id']:r for r in (load(f) for f in (run/(source+'-serial')).glob('*.json'))}
  compared=[(r,original_by_id[r['id']]) for r in rows]
  out[source]={'distinct_queries':len(set(r['id'] for r in rows)),'requests':len(rows),'answerable':metric['answerable'],
    'complete_answers':metric['full_grounded_answers'],'insufficient':metric['insufficient'],'justified_abstentions':metric['justified_abstentions'],
    'opened_verified_in_scope':metric['original_links_opened_and_body_scope_verified'],'supported_citations':metric['all_cited_lines_supported_by_own_original'],
    'required_fact_units':metric['required_fact_units_on_positives'],'required_fact_units_cited':metric['required_fact_units_cited_on_positives'],'modes':metric['modes'],
    'compared_to_serial':{'equal_ordered_source_lists':sum(r['opened_source_ids']==o['opened_source_ids'] for r,o in compared),'mode_changes':sum(r['mode']!=o['mode'] for r,o in compared)},
    'loopback_answer_ms_nearest_rank':{label:times[math.ceil(q*len(times))-1] for label,q in [('p50',.5),('p95',.95),('p99',.99)]},
    'by_family':{f:{k:z[k] for k in ('total','answerable','full_grounded_answers','insufficient','justified_abstentions')} for f,z in record['by_family'].items()}}
 summary['phases'][phase]={'clients':data['concurrency'],'sources':out}
for name,base in [('rag_facts','rag_final.json'),('validation_pilot_rag_facts','validation_pilot/rag-final.json'),('rag_unseen','rag_unseen_final.json'),('rag_confirmation','rag_confirmation_final.json')]:
 original=load(root/'results'/base);now=load(run/(name+'.json'));assert now['fixture_sha256']==original['fixture_sha256']
 keys=('answerable','all_facts_with_expected_citation','insufficient','explicit_no_evidence','required_facts','required_facts_with_citation','originals_opened_in_scope')
 summary['frozen_regressions'][name]={'fixture_sha256':original['fixture_sha256'],'previous':{k:original['summary'][k] for k in keys},'current':{k:now['summary'][k] for k in keys}}
summary['notes']='Full high-concurrency phase contains each VK query once across eight clients and the single personal query once. The separate repeated panel supplies eight simultaneous requests for that personal case. Previous fixture scoring is the project’s registered fact-and-source checker.'
(run/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
for name,d in summary['frozen_regressions'].items():
 keys=('all_facts_with_expected_citation','explicit_no_evidence'); print(name,[(d['previous'][k],d['current'][k]) for k in keys])
for phase,d in summary['phases'].items():
 print(phase,[(s,val['complete_answers'],val['justified_abstentions'],val['supported_citations'],val['compared_to_serial'],val['loopback_answer_ms_nearest_rank']) for s,val in d['sources'].items()])
