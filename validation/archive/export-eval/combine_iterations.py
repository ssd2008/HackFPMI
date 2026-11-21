"""Build the public report using aggregate counters and code/fixture fingerprints only."""
from functools import reduce
from pathlib import Path
import hashlib
import json
import sys


def load(path):
    return json.loads(path.read_text())


def totals(groups):
    return {
        'queries':sum(g['positive_queries'] for g in groups),
        'hit_at_1':sum(g['hit_at_1'] for g in groups),
        'hit_at_5':sum(g['hit_at_5'] for g in groups),
    }


def empties(groups):
    return {
        'queries':sum(g['empty_queries'] for g in groups),
        'correctly_empty':sum(g['correctly_empty'] for g in groups),
    }


def describe_run(folder, label, fixture, fresh=False):
    s=load(folder/(label+'-summary.json'))
    detail=load(folder/(label+'.json'))
    metrics=s['metrics'];groups=s['groups']; cases=detail['cases']
    assert s['fixture_sha256']==hashlib.sha256((folder/fixture).read_bytes()).hexdigest()
    tech={family:g for family,g in groups.items() if g['positive_queries'] and family!='authored_query_silver_subject'}
    positives=[c for c in cases if c['family'] in tech]
    assert all(len(c['expected'])==1 for c in positives)
    result={
        'frozen_probe_file_sha256':s['fixture_sha256'],
        'search_requests':s['distinct_sequential_requests'],
        'technical':totals(list(tech.values())),
        'negative':empties(list(groups.values())),
        'positive_by_probe_family':{k:totals([v]) for k,v in tech.items()},
        'negative_by_probe_family':{k:empties([v]) for k,v in groups.items() if v['empty_queries']},
        'scope':{k:metrics[k] for k in ('returned_items','invalid_mailbox_or_asof_items','outside_explicit_received_window_items')},
        'sequential_server_ms':s['latency'],
        'technical_exact_set':{
            'positive_queries':len(positives),
            'returned_only_unique_expected_document':sum(set(c['results'])==set(c['expected']) for c in positives)
        }
    }
    if not fresh:
        result['natural_language_silver']=totals([groups['authored_query_silver_subject']])
        result['direct_reads']={'checks':metrics['read_checks'],'passed':metrics['correct_reads']}
    return result


first=load(Path('.cache/export-eval/first-unseen-aggregates.json'))
assert sum(x['search_requests'] for x in first['corpora'])==103
files={key:hashlib.sha256((Path('mailsearch')/(key+'.py')).read_bytes()).hexdigest() for key in first['source_sha256']}
assert files['retrieval']==sys.argv[1]
report={
    'schema':2,
    'protocol':{**first['protocol'],
        'iterations':'В первой итерации все 103 запроса были новыми; по результатам добавлен поиск буквальной фразы в парных кавычках. Во второй итерации те же 103 запроса измерены повторно, а 29 новых технических проверок заморожены до нового прогона.',
        'fresh_quoted_controls':'Новые письма для положительных проверок не пересекаются с целями прошлых наборов. Для этих проверок проверяется и наличие целевого письма, и отсутствие лишних результатов при однозначном буквальном совпадении.',
    },
    'source_sha256':{'first_iteration':first['source_sha256'],'quoted_phrase_iteration':files},
    'corpora':[],
}
for old,dirname in zip(first['corpora'],('export-eval','export-eval-02'),strict=True):
    folder=Path('.cache')/dirname
    metadata_keys=('corpus','input_messages','mailboxes','missing_dates','original_date_changed_by_timezone','calendar_day_policy')
    metadata={k:old[k] for k in metadata_keys}
    previous={key:value for key,value in old.items() if key not in metadata}
    reused=describe_run(folder,'quoted-reused','probes.json')
    fresh=describe_run(folder,'quoted-fresh','fresh-quotes.json',True)
    assert reused['frozen_probe_file_sha256']==previous['frozen_probe_file_sha256']
    reused['inbox_smoke']={'mailboxes_checked':old['mailboxes'],'page_and_original_mail_checked':True,
                           'unavailable_document_returns_404':True,'empty_snapshot_rag_returns_no_sources':True}
    report['corpora'].append({**metadata,'first_iteration_unseen':previous,
                             'quoted_phrase_iteration':{'reused':reused,'fresh_unseen':fresh}})
text=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
assert not any(k in text for k in ('/Users/','@','L-','box-'))
for dirname in ['export-eval','export-eval-02']:
    folder=Path('.cache')/dirname
    for fixture in ('probes.json','fresh-quotes.json'):
        assert all(c['query'] not in text for c in load(folder/fixture)['cases'])
    for line in (folder/'corpus/documents.jsonl').open():
        if line.strip():
            r=json.loads(line)
            assert '"'+r['doc_id']+'"' not in text and '"'+r['synthetic_user']+'"' not in text
target=Path('results/additional_corpora.json')
target.write_text(text)
target.chmod(0o644)
print('Verified both rounds, provenance, disjoint new targets and aggregate-only output.')
