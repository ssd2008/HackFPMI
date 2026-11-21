from argparse import Namespace
from unittest.mock import patch
from pathlib import Path
import json, sys

root = Path.cwd()
sys.path.insert(0,str(root/'.cache/round4-answer'))
from tools import check_four_metrics as f
from tools.validate_rankings import prepare_tasks

ids = ['single','many','nothing']
queries={q:{'query_id':q,'synthetic_user':'example','query':'example '+q,'query_date':'2023-01-02'} for q in ids}
docs={'one':('example','2023-01-01'),'two':('example','2023-01-01'),'future':('elsewhere','2024-01-01')}
labels={'single':{'one':2},'many':{'two':2,'one':1},'nothing':{}}
meta={q:{'query_id':q,'family_id':q,'source':'sample','split':'dev','state':'verified_empty' if q=='nothing' else 'positive','task_type':kind,'target_ids':['one'] if q=='single' else [],'receipt_window':['2023-01-01','2023-01-01']} for q,kind in zip(ids,['known_item','collection','empty'])}
w=f.Workload('fixture',docs,queries,labels,meta,prepare_tasks(list(meta.values()),queries,labels),{'documents.jsonl':'fake'})
a={'single':('future','one'),'many':(),'nothing':('two',)};b={'single':('one',),'many':('two','one'),'nothing':()}
def answer(url,query,k):
    assert k==10
    return (a if url=='A' else b)[query['query_id']],0.25,0.5
with patch.object(f,'fetch',side_effect=answer) as http, patch.object(f,'check_service_corpus',return_value='sha256'):
    got=f.evaluate(w,{'before':'A','after':'B'},Namespace(clients=[1,8],pairs=2,min_samples=4,seed=100),0)
assert http.call_count==60 and got['within_version_changed_rankings']=={'before':0,'after':0}
for clients,orders in [('1',[['before','after'],['after','before']]),('8',[['after','before'],['before','after']])]:
    d=got['by_clients'][clients];assert d['requests_per_side']==12 and [e['order'] for e in d['pairs']]==orders
    vals=got['four_metrics_by_clients'][clients]
    assert vals['recall_at_10_positive']=={'before':0.5,'after':1,'delta_after_minus_before':0.5}
    assert vals['mrr_at_10_single']=={'before':0.5,'after':1,'delta_after_minus_before':0.5}
    assert vals['ndcg_at_10_positive']['before']==0.31546488 and vals['client_p99_ms']['before']>=0
assert got['quality']['before']['false_nonempty']==got['quality']['before']['false_empty']==1
assert got['quality']['after']['false_nonempty']==got['quality']['after']['false_empty']==0
assert got['scope_across_distinct_query_rankings']['before']['future']==got['scope_across_distinct_query_rankings']['before']['foreign_mailbox']==1
assert got['scope_across_distinct_query_rankings']['before']['outside_requested_window']==1
assert got['scope_across_distinct_query_rankings']['before']['distinct_query_rankings']==3
r,sc={},{}
with patch.object(f,'fetch',side_effect=[(['one'],1,1),(['future'],1,1),(['one'],1,1)]):
    assert f.timed_series('A',w,['single']*3,1,r,sc)[2]==1
    assert len(sc)==2 and sum(c['future'] for c in sc.values())==1
with patch.object(f,'fetch',return_value=(['one','one'],1,1)):
    try: f.timed_series('A',w,['single'],1,{},{});raise AssertionError('duplicate accepted')
    except ValueError: pass
print('Controlled transport validated. No HTTP connections made.')

pool=Path('/Users/kirill/Downloads/HackFPMI-mail-OOD')
for work,path in [(f.official(root/'data'),root/'results/public_dev.json')]+[(f.panel(pool,s),root/f'.cache/ood-mail/panels/{s}_dev/after/rankings.json') for s in ['vk','personal']]:
    saved=json.loads(path.read_bytes()); q=f.quality(work,{qid:tuple(row['ranked_ids'][:10]) for qid,row in saved['queries'].items()}); prev=saved['quality_first_pass']['10']
    if work.name=='official_vk':
        verified=[(q[new],prev[old]) for old,new in [('recall','recall_at_10_positive'),('ndcg','ndcg_at_10_positive'),('mrr_single','mrr_at_10_single')]]
    else:
        aggregate=json.loads((root/'results/ood_mail_retrieval.json').read_bytes())['datasets'][work.name]['panel']['primary']['metrics']
        verified=[(q[new],aggregate[old]['after']) for old,new in [('known_recall_at_10','recall_at_10_positive'),('ndcg_at_10','ndcg_at_10_positive'),('mrr_at_10','author_target_mrr_at_10')]]
        per_single=[saved['queries'][qid]['scores']['10']['mrr_single'] for qid in work.queries if saved['queries'][qid]['relevant']==1]
        verified.append((q['mrr_at_10_single'],sum(per_single)/len(per_single)))
    for measured,expected in verified: assert abs(measured-expected)<0.0000006,(work.name,measured,expected)
    print(work.name,'stored aggregate metrics reproduced; singleton / known-item:',q['mrr_queries'],q['author_target_queries'])
