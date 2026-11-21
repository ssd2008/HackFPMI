"""Compare complete ordered lists; only aggregates and canonical digests leave the cache."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

H = Path.cwd()
D = H / '.cache/round4'
B = '61bae2d95ac6bf32a9bd59dbc1056173439bfae3'
C = 'd18d1a06f5fc07b4ed84dfeeb6ac587b099d2a9d'
A = 'b5252c144fc709168c96f8957d97c1d9ee5353af'
OUT = D / 'd18-rank-check'
OUT.mkdir(exist_ok=True, mode=0o700)
LOCAL = {B:D/'prior-search', C:D/'prior-search-d18'}


def read(p):
    return json.loads(p.read_bytes())


def rows(p):
    return [json.loads(line) for line in p.read_bytes().splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def filehash(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compare(x, y):
    assert set(x) == set(y)
    different = sum(x[k] != y[k] for k in x)
    return {'queries': len(x), 'identical_order': len(x) - different, 'different': different,
            'previous_sha256': digest(x), 'current_sha256': digest(y)}


def core():
    return {'previous_revision': B, 'current_revision': C,
            'method': 'Exact equality of the full returned top-10 ID sequence for every query, including empty lists. Canonical SHA-256 per channel maps the query ID to the SHA-256 of its ordered ID list; JSON keys are sorted, ASCII escaped, with compact separators.',
            'sets': {}}


def publish(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.chmod(temp, 0o644)
    temp.replace(path)


def fallback_ranks(rev):
    code = LOCAL[rev] / rev / 'engine'
    sys.path.insert(0, str(code))
    from mailsearch import app
    from mailsearch.store import MailStore
    from mailsearch.retrieval import SearchEngine
    for name in ('app','retrieval','store'):
        module = sys.modules['mailsearch.' + name]
        src = Path(module.__file__).resolve()
        assert src.parent == (code / 'mailsearch').resolve()
        assert src.read_bytes() == subprocess.check_output(['git','show',rev+':mailsearch/'+name+'.py'])
    all_sets = {}
    for name, folder, docs_path, fixtures in (
        ('vk', H/'tests/fixtures', H/'data/documents.jsonl', [('probes','search_probes.json'),('probes_extra','search_probes_extra.json')]),
        ('export_1', H/'.cache/export-eval', H/'.cache/export-eval/corpus/documents.jsonl', [('original','probes.json'),('quote','fresh-quotes.json')]),
        ('export_2', H/'.cache/export-eval-02', H/'.cache/export-eval-02/corpus/documents.jsonl', [('original','probes.json'),('quote','fresh-quotes.json')]),
        ('wiring', H/'.cache/iteration/separate-corpus', H/'.cache/iteration/separate-corpus/documents.jsonl', [('cases','queries_dev.jsonl')]),
    ):
        docs=rows(docs_path); catalog={d['doc_id']:(d['synthetic_user'],d['doc_dt'])for d in docs}
        app.app.state.store = MailStore(docs)
        app.app.state.search_engine = SearchEngine(docs,H/'.cache/mailsearch')
        for suite, file in fixtures:
            p=folder/file
            if p.suffix=='.json':
                fixture=read(p); assert fixture['documents_sha256']==filehash(docs_path)
                qs=[(c['id'],c['mailbox'],c['date'],c['query'])for c in fixture['cases']]
            else:
                qs=[(q['query_id'],q['synthetic_user'],q['query_date'],q['query'])for q in rows(p)]
            seen={}
            for qid,box,day,text in qs:
                response=app.search(app.SearchRequest(mailbox=box,date=day,query=text,limit=10))
                payload=json.loads(response.body) if hasattr(response,'body') else response
                ids=[i['id'] for i in payload['items']]
                assert len(ids)==len(set(ids))<=10 and qid not in seen
                assert all(catalog[d][0]==box and (not catalog[d][1] or catalog[d][1]<=day) for d in ids)
                seen[qid]=digest(ids)
            all_sets[name+'_'+suite]={'snapshot':filehash(docs_path),'fixture':filehash(p),'ranks':seen}
    (OUT/(rev+'.json')).write_text(json.dumps(all_sets,separators=(',',':')))


def prior():
    new = read(LOCAL[C]/'final.json')
    old = read(LOCAL[B]/'final-verified.json')
    assert old['revisions']['current']==B and new['revisions']['current']==C and old['revisions']['baseline']==new['revisions']['baseline']
    for part in ['public_variants','vk_probes','private_probes','wiring_smoke']:
        print(part, 'aggregates match', old[part]==new[part])
    assert old['pilot']['groups']==new['pilot']['groups']
    equiv=core()
    for key in ['paraphrased','typos','no_date','spelling_stress','pilot']:
        s=[read(LOCAL[r]/r/(key+'.json'))for r in (B,C)]
        assert s[0]['dataset']['sha256']==s[1]['dataset']['sha256']
        encoded=[{q:digest(row['ranked_ids'])for q,row in obj['queries'].items()} for obj in s]
        equiv['sets'][key]=compare(*encoded)
    local=[read(OUT/(r+'.json')) for r in (B,C)]
    assert local[0].keys()==local[1].keys()
    for key,x in local[0].items():
        y=local[1][key];assert x['snapshot']==y['snapshot'] and x['fixture']==y['fixture']
        equiv['sets'][key]=compare(x['ranks'],y['ranks'])
    assert sum(x['queries'] for x in equiv['sets'].values())==501
    new['previous_release_rank_equivalence']=equiv
    publish(H/'results/ood_round4_prior_search.json',new)
    print('prior',[(k,v['queries'],v['different'])for k,v in equiv['sets'].items()])


def ext(group):
    prev=D/('transfer-final-'+group)
    now=D/('transfer-d18-'+group)
    a,b=read(prev/'freeze.json'),read(now/'freeze.json')
    assert a['engines']=={'baseline':A,'current':B} and b['engines']=={'baseline':A,'current':C}
    assert a['inputs']==b['inputs'] and a['selection']==b['selection'] and a['cutoff']==b['cutoff']==10
    published=read(now/'report.json'); assert published['protocol']==b
    equiv=core()
    for name, meta in a['inputs'].items():
        sources=[]
        for loc,rev, protocol in [(prev,B,a),(now,C,b)]:
            raw=read(loc/name/(rev+'.json'))
            assert raw['dataset']==name and raw['engine_revision']==rev and raw['freeze_sha256']==filehash(loc/'freeze.json')
            assert set(raw['ranked_ids'])==set(meta['queries'])
            sources.append(raw)
        old,new=sources; assert old['vectors_sha256']==new['vectors_sha256']
        directly=[]
        for ch,n in meta['queries'].items():
            xr, yr = [s['ranked_ids'][ch] for s in sources]
            assert len(xr)==len(yr)==n and set(xr)==set(yr)
            direct=sum(xr[q]!=yr[q]for q in xr)
            x,y=[{k:digest(order)for k,order in r.items()}for r in (xr,yr)]
            result=compare(x,y)
            assert result['different']==direct
            equiv['sets'][name+'/'+ch]=result
            directly.append(direct)
        print(name,directly)
    published['previous_release_rank_equivalence']=equiv
    publish(H/('results/ood_round4_transfer_'+group+'.json'),published)


if __name__=='__main__':
    if sys.argv[1]=='worker': fallback_ranks(sys.argv[2])
    elif sys.argv[1]=='prior': prior()
    else: ext(sys.argv[1])
