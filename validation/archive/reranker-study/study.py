"""Frozen-candidate cross-encoder study. Run from repository root using the project's venv."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from mailsearch.dates import parse_date_scope
from search_bench.ranking import query_text
from benchmark import metrics as query_metrics

MODEL_ID = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
REVISION = '1427fd652930e4ba29e8149678df786c240d8825'
EXPECTED_WEIGHT_SHA = '5daeca2481a76b5976a2bdc32f0a78532b6716da4f8cd3ff59460ef8d2f359b4'
MAXLEN = 256
TOPS = [10, 32, 50]
KS = [1, 5, 10, 20, 50]
FILES = {}
for key in ('dev','paraphrases'):
    FILES[key] = {
        'mvp': ROOT / f'search_bench/results/mvp-{key}.json',
        'e5': ROOT / f'search_bench/results/{key}.json'
    }


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(name, value):
    dest = HERE / name
    temp = dest.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    temp.replace(dest)


def hash_files():
    hashes = {}
    for group in FILES.values():
        for path in group.values():
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def rrf(rankings, k=60):
    scores = defaultdict(float)
    for order in rankings:
        for rank, doc_id in enumerate(order, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    # Dict insertion order resolves exact ties, giving the first retriever priority.
    return sorted(scores, key=scores.__getitem__, reverse=True)


def load_frozen():
    documents = {d['doc_id']: d for d in map(json.loads,(ROOT/'data/documents.jsonl').open())}
    groups = {}
    for label, paths in FILES.items():
        mvp = load_json(paths['mvp'])['methods']['mvp-api']['per_query']
        e5 = load_json(paths['e5'])['methods']['e5-date']['per_query']
        rows = []
        for m,e in zip(mvp,e5,strict=True):
            fields = ['query_id','query','query_date','synthetic_user']
            assert [m[f] for f in fields] == [e[f] for f in fields]
            scope = parse_date_scope(m['query'],date.fromisoformat(m['query_date']))
            a,b = m['doc_ids'], e['doc_ids']
            assert max(len(a),len(b)) <= 50
            assert len(a) == len(set(a)) and len(b) == len(set(b))
            union = rrf([a,b])
            for did in union:
                d = documents[did]
                assert d['synthetic_user'] == m['synthetic_user']
                assert d['doc_dt'] <= m['query_date']
                assert not scope.start or date.fromisoformat(d['doc_dt']) >= scope.start
                assert not scope.end or date.fromisoformat(d['doc_dt']) <= scope.end
            rows.append({'id':m['query_id'],'query':m['query'],'as_of':m['query_date'],'box':m['synthetic_user'],
                         'text':query_text(m['query'],scope),'mvp':a,'e5':b,'union':union})
        assert len(rows)==99
        groups[label] = rows
    return documents,groups


class Encoder:
    def __init__(self, device):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        torch.set_num_threads(4)
        self.device = device
        args = {'local_files_only':True,'trust_remote_code':False}
        self.tokenizer = AutoTokenizer.from_pretrained(HERE/'model', **args)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            HERE/'model', use_safetensors=True, dtype=torch.float32 if device=='cpu' else torch.float16,
            **args
        ).to(device).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

    def score(self, query, texts):
        import numpy as np
        if not texts:
            return []
        result = []
        for start in range(0,len(texts),32):
            part = texts[start:start+32]
            tokens = self.tokenizer([query]*len(part), part, max_length=MAXLEN,padding=True,
                                    truncation='only_second', pad_to_multiple_of=8,return_tensors='pt')
            with self.torch.inference_mode():
                logits = self.model(**tokens.to(self.device)).logits.squeeze(-1)
                result.extend(logits.float().cpu().tolist())  # synchronizes the accelerator
        assert len(result)==len(texts) and np.isfinite(result).all()
        return result


def obtain_scores(device):
    docs,groups = load_frozen()
    with (HERE/'model/model.safetensors').open('rb') as source:
        assert hashlib.file_digest(source,'sha256').hexdigest() == EXPECTED_WEIGHT_SHA
    started=perf_counter()
    encoder=Encoder(device)
    load_seconds=perf_counter()-started
    result={}
    for label,rows in groups.items():
        t=perf_counter()
        scored={}
        for row in rows:
            ids=row['union']
            logits=encoder.score(row['text'],[docs[key]['doc'] for key in ids])
            scored[row['id']] = dict(zip(ids,logits,strict=True))
        result[label]=scored
        print('scores',label,len(scored),sum(map(len,scored.values())),round(perf_counter()-t,2),flush=True)
    out={'model':MODEL_ID,'revision':REVISION,'weights_sha256':EXPECTED_WEIGHT_SHA,'max_pair_tokens':MAXLEN,
         'parameters':encoder.parameter_count,'device':device,'load_seconds_including_imports':load_seconds,
         'frozen_files_sha256':hash_files(),'scores':result}
    dump('scores.json',out)


def rank_variants(row, scores):
    pools = {key: row[key][:50] for key in ('mvp','e5','union')}
    rankings = {key: ids for key,ids in pools.items()}
    for pool, ids in pools.items():
        for top in TOPS:
            head=ids[:top]
            reranked=sorted(head, key=scores.__getitem__, reverse=True)
            rankings[f'{pool}-ce{top}'] = reranked+ids[top:]
            rankings[f'{pool}-rrf-ce{top}'] = rrf([head,reranked])+ids[top:]
    return rankings


def quality():
    _,groups=load_frozen()
    cached=load_json(HERE/'scores.json')
    assert hash_files()==cached['frozen_files_sha256']
    gold=defaultdict(dict)
    for r in map(json.loads,(ROOT/'data/qrels_dev.jsonl').open()):
        gold[r['query_id']][r['doc_id']]=r['relevance']
    report={}
    for label,rows in groups.items():
        vals=defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        all_rankings={}
        for row in rows:
            ranks=rank_variants(row,cached['scores'][label][row['id']])
            all_rankings[row['id']]=ranks
            for name,ids in ranks.items():
                for k in KS:
                    for metric,val in query_metrics(ids,gold[row['id']],k).items():
                        vals[name][str(k)][metric].append(val)
        aggregated={}
        for name,by_k in vals.items():
            current=by_k['10']['ndcg']
            base=vals['mvp']['10']['ndcg']
            diffs=[x-y for x,y in zip(current,base)]
            group_qids=Counter(1 if v>1e-9 else -1 if v < -1e-9 else 0 for v in diffs)
            aggregated[name]={'metrics':{k:{m:statistics.mean(v) for m,v in met.items()} for k,met in by_k.items()},
                              'paired_ndcg10_vs_mvp':{tag:group_qids[n] for tag,n in (
                                  ('up',1),('same',0),('down',-1))}}
        report[label]={'methods':aggregated,'per_query':all_rankings}
        print(label)
        for name in ('mvp','e5','union','mvp-ce32','e5-ce32','union-ce32','union-rrf-ce32','union-ce50'):
            print(name,aggregated[name]['metrics']['10'],aggregated[name]['paired_ndcg10_vs_mvp'])
    dump('quality.json',report)


def percentiles(values):
    seq=sorted(values)
    return {f'p{n}':round(seq[math.ceil(len(seq)*n/100)-1],3) for n in (50,95,99,100)}


def latency(device):
    docs,groups=load_frozen()
    print('loading',device,flush=True)
    start=perf_counter();engine=Encoder(device);load=perf_counter()-start
    rows=[(label,row) for label,ls in groups.items() for row in ls]
    random.Random(1741).shuffle(rows)
    start=perf_counter();engine.score(rows[0][1]['text'],[docs[did]['doc'] for did in rows[0][1]['union'][:16]])
    cold_ms=(perf_counter()-start)*1000
    print('model startup seconds',load,'first rerank ms',cold_ms,flush=True)
    report={'device':device,'cpu_threads':4,'platform':platform.platform(),'parameters':engine.parameter_count,
            'precision':'float32' if device=='cpu' else 'float16','max_pair_tokens':MAXLEN,
            'load_seconds_including_imports':load,'first_rerank_ms':cold_ms,
            'first_rerank_pairs':min(len(rows[0][1]['union']),16),'policies':{}}
    # Same frozen union, fixed 32 pair microbatch, loopback/backend excluded.
    for top in TOPS:
        timed=[]
        hash_hits=[]
        lengths=[]
        pair_count=0
        cache={}
        for label,row in rows:
            t=perf_counter()
            ids=row['union'][:top]
            if len(ids)>1:
                scores=engine.score(row['text'],[docs[did]['doc'] for did in ids])
                ordered=[did for did,score in sorted(zip(ids,scores),key=lambda p:p[1],reverse=True)]
            else:
                ordered=ids.copy()
            assert len(ordered)==len(ids)
            timed.append((perf_counter()-t)*1000)
            lengths.append(len(ids))
            pair_count+=len(ids) if len(ids)>1 else 0
            cache[(row['box'],row['query'],row['as_of'],tuple(ids))]=ordered
        # Microbenchmark a best-case exact result-cache hit, excluding first-stage work,
        # network, and everything except the tuple-key lookup and list copy.
        for _,row in rows:
            t=perf_counter()
            list(cache[(row['box'],row['query'],row['as_of'],tuple(row['union'][:top]))])
            hash_hits.append((perf_counter()-t)*1000)
        report['policies'][str(top)]={'end_to_end_rerank_ms':percentiles(timed),
            'positive_pool_ms':percentiles([v for v,n in zip(timed,lengths) if n>1]),
            'full_pool_ms':percentiles([v for v,n in zip(timed,lengths) if n==top]),
            'cache_hit_python_only_ms':percentiles(hash_hits),'requests':len(rows),
            'requests_with_full_pool':lengths.count(top),'model_calls_skipped':sum(n<=1 for n in lengths),
            'pairs_scored':pair_count,'median_candidates':statistics.median(lengths),'exact_cache_keys':len(cache)}
        print('timing',device,top,report['policies'][str(top)],flush=True)
        dump(f'latency-{device}.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['score','quality','latency'])
    p.add_argument('--device',choices=['cpu','mps'],default='mps')
    a=p.parse_args()
    if a.task=='quality': quality()
    elif a.task=='score': obtain_scores(a.device)
    else: latency(a.device)
