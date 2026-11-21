"""Create an origin-blinded pool and attach the independent, previously frozen judgments."""
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import random
import re
import sys

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
sys.path.insert(0,str(ROOT))
from benchmark import fetch, metrics
from search_bench.ranking import Search
from mailsearch.dense import _index_name
FIX = ROOT / 'tests/fixtures/validation_pilot'
def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]
def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def dump_lines(path, value):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in value))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

freeze = json.loads((WORK/'freeze.json').read_bytes())
assert all(sha(FIX/f)==signature for f,signature in freeze['files'].items())
docs = read(ROOT/'data/documents.jsonl')
dby = {d['doc_id']:d for d in docs}
qs = {q['query_id']:q for q in read(FIX/'queries.jsonl')}
ts = {t['query_id']:t for t in read(FIX/'tasks.jsonl')}
labels = {qid:{} for qid in qs}
for row in read(FIX/'qrels.jsonl'):
    labels[row['query_id']][row['doc_id']] = row['relevance']
ranked={}
for name,url in (('before','http://127.0.0.1:18773'),('after','http://127.0.0.1:8766')):
    r = json.loads((WORK/(name+'.json')).read_text())['queries']
    ranked[name]={qid:v['ranked_ids'] for qid,v in r.items()}
    for qid, t in ts.items():
        if t['task_type']=='collection':
            deep, _, _ = fetch(url, qs[qid], 20)
            assert deep[:10]==ranked[name][qid], (qid,name)
            ranked[name][qid]=deep
# Reuse only the pre-existing public-corpus index, in a study-specific cache.
public_vectors=ROOT/'.cache/mailsearch'/_index_name(docs)
private_cache=WORK/'public_vectors'
private_cache.mkdir(exist_ok=True)
link=private_cache/public_vectors.name
if public_vectors.is_file() and not link.exists():
    link.symlink_to(public_vectors)
options = ['date-only','bm25-date','e5-date']
engine=Search(docs,options,private_cache)
for method in options:
    ranked[method]={}
    for qid,q in qs.items():
        depth=20 if ts[qid]['task_type']=='collection' else 10
        ranked[method][qid]=engine.rank(method,q['query'],q['synthetic_user'],date.fromisoformat(q['query_date']),depth)
dump(WORK/'pool-origins.json',ranked)
# Deterministic corpus-only distractors, using surface facts from the registered goals.
near = [
    r'гурьев|муниципальн', r'пневмо|гайковерт|гайковёрт', r'пункт.*выдачи|возврат',
    r'муниципальн|пени', r'резьб|метчик', r'продавец отменил|отмен[её]н',
    r'микробиолог|исследован', r'этап производства|поле объекта', r'размораж|стекол|стёкол',
    r'недоступ', r'медико.санитарная|фсвок', r'код|билет|посадо'
]
pooled=[]; paired=[]; per_q={}; tally=Counter(); provenances=Counter()
for i,(qid,q) in enumerate(qs.items()):
    lab=labels[qid];t=ts[qid]; dset=set(lab)
    assert all(set(byq[qid])<=dset for byq in ranked.values())
    rng=random.Random(482012+i)
    related=[did for did in sorted(lab) if lab[did]==0 and re.search(near[i],dby[did]['doc'],re.I)]
    if t.get('received_window'):
        start,end=t['received_window']
        related.sort(key=lambda did:not start<=(dby[did]['doc_dt'] or '0')<=end)
        in_window=[did for did in related if start<=(dby[did]['doc_dt'] or '0')<=end]
        related=in_window or related
    independent=rng.sample(related,min(4,len(related))) + rng.sample(sorted(lab),min(3,len(lab)))
    gold={did for did,g in lab.items() if g>0}
    origins={'before','after','bm25-date','e5-date'}
    union=gold|set(independent)|set.union(*(set(ranked[n][qid]) for n in origins))
    shuffled=sorted(union)
    rng.shuffle(shuffled)
    basecov=set(ranked['before'][qid][:10])|set(ranked['after'][qid][:10])
    cnt=Counter(lab[did] for did in shuffled)
    tally.update(cnt)
    for j,did in enumerate(shuffled):
        # This view can be re-judged without revealing rank, origin or the frozen grade.
        blinded='vk-card-'+hashlib.sha256(f'validation-pilot:20261121:{qid}:{did}'.encode()).hexdigest()[:16]
        pooled.append({'query_id':qid,'query':q['query'],'query_date':q['query_date'],'card_id':blinded,'received_date':dby[did]['doc_dt'],'text':dby[did]['doc']})
        paired.append({'query_id':qid,'card_id':blinded,'doc_id':did,'relevance':lab[did]})
    per_q[qid]={
        'split':t['split'],'task_type':t['task_type'],'depth':20 if t['task_type']=='collection' else 10,
        'visible':len(dset),'registered_positives':len(gold),'unique_pool':len(shuffled),'pool_grades':dict(sorted(cnt.items())),
        'nonranked_positive_additions':len(gold-set().union(*(set(ranked[n][qid]) for n in origins))),
        'complete_api_top10_unique':len(basecov),'exhaustive_coverage':sum(d in lab for d in shuffled)/len(shuffled),
        'lengths':{name:len(ranked[name][qid]) for name in sorted(origins)},'n_independent_visible':len(set(independent)),
    }
    for n in origins:provenances[n]+=len(ranked[n][qid])
dump_lines(WORK/'pool-blinded.jsonl',pooled)
dump_lines(WORK/'pool-graded.jsonl',paired)
baselines={}
for method in options:
    bysplit={}
    for split in ('dev','blind'):
        ids=[qid for qid,t in ts.items() if t['split']==split and t['state']=='positive']
        vals=[metrics(ranked[method][qid],labels[qid],10) for qid in ids]
        bysplit[split]={'positive_n':len(vals),'ndcg10':sum(row['ndcg'] for row in vals)/len(vals),'recall10':sum(row['recall'] for row in vals)/len(vals)}
    baselines[method]=bysplit
audit={'freeze':freeze['freeze'],'version':{'before':'5fe3f5d','after_ranker':'d747e72d'},'seed':482012,
       'source':'VK synthetic corpus only','corpus_sha256':sha(ROOT/'data/documents.jsonl'),
       'judgments':'Exhaustive registered agent rules. No human annotation or human agreement measured.',
       'judgment_sequence':'The targets, criteria, grading outputs and query files were frozen before retrieving either API or offline runs. Shuffled pool grade uses this precomputed independent mapping.',
       'depth':'10, or 20 for all 5 collections. The second API request for pooling used the frozen configurations, agreed exactly on primary top-10 and did not update judgments.',
       'pool_sources':dict(provenances),'unique_query_document_pairs':len(pooled),'pool_grades':dict(sorted(tally.items())),
       'complete_visible_qrels':sum(map(len,labels.values())),'frozen_corpus_qrels_grades':dict(sorted(Counter(g for m in labels.values() for g in m.values()).items())),
       'independent_positives_exclusive':sum(row['nonranked_positive_additions'] for row in per_q.values()),
       'public_model_index_name':public_vectors.name,'offline_rankers':baselines,'per_query':per_q}
dump(WORK/'pool-audit.json',audit)
print('done',len(pooled),tally,baselines,flush=True)
