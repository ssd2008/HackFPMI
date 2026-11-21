import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
FIX=ROOT/'tests/fixtures/validation_pilot'
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    return [json.loads(s) for s in p.read_text().splitlines()]
hashes=json.loads((ROOT/'results/validation_pilot/freeze.json').read_text())['files']
assert all(digest(FIX/file)==v for file,v in hashes.items())

fact_patterns={
    'vk-pilot-01':[r'(?<!\d)504[.,]75(?!\d)|(?<!\d)504\s+руб\w*\s+75\s+коп\w*'],
    'vk-pilot-02':[r'(?<!\d)6[ \u00a0\u202f]?655[.,]17\s*(?:RUB\b|₽|руб\.|рубл(?:я|ей|ь|ях|ями)\b)|(?<!\d)6[ \u00a0\u202f]?655\s+руб\w*\s+17\s+коп\w*'],
    'vk-pilot-03':[r'(?<!\d)457(?!\d)',
                   r'(?:покаж\w*|показ\w*|предъяв\w*|предостав\w*)[^.\n]{0,90}?штрих[ -]?код|штрих[ -]?код[^.\n]{0,90}?(?:покаж\w*|показ\w*|предъяв\w*)'],
    'vk-pilot-07':[r'(?:иммуногенетик\w*)(?:\s|[«»:=,–—-]|(?:код\w*)|(?:присвоен\w*)|(?:имеет)|(?:это)|(?:соответству\w*)|(?:значени\w*)){0,32}\b416\b|\b416\b(?:\s|[«»:=,–—-]|(?:код)|(?:для)|(?:соответству\w*)){1,32}иммуногенетик\w*|^\s*(?:код\s*)?[:—–-]?\s*416(?=\s*[.!]?\s*(?:\[\d+\]|$))'],
    'vk-pilot-08':[r'евдокимов(?:а|ой|у|е)\s+вероник(?:а|е|и|ой|у)\s+харитоновн(?:а|е|ы|ой|у)|вероник(?:а|е|и|ой|у)\s+харитоновн(?:а|е|ы|ой|у)\s+евдокимов(?:а|ой|у|е)'],
}
# The 504.75 source does not name a currency; no currency may be inferred from a numeric hit.
fact_patterns['vk-pilot-01']=[r'(?<!\d)504[.,]75(?!\d)']
qs={q['query_id']:q for q in read(FIX/'queries.jsonl')}
docs={d['doc_id']:d for d in read(ROOT/'data/documents.jsonl')}
cases=[]
for t in read(FIX/'tasks.jsonl'):
    if t['task_type'] not in {'known_item','empty'}:
        continue
    q=qs[t['query_id']]
    case={'id':q['query_id'],'split':t['split'],'mailbox':q['synthetic_user'],'date':q['query_date'],
          'query':q['query'],'sources':t['target_ids'],'facts':fact_patterns.get(q['query_id'],[]),
          'expected_outcome':'supported_answer' if t['task_type']=='known_item' else 'justified_abstention',
          'expected':t.get('required_fact') or t['empty_proof']}
    if t['task_type']=='known_item':
        case['evidence_quote']=t['evidence_quote']
    if q['query_id']=='vk-pilot-03':
        case['additional_evidence_quote']='Чтобы получить возвраты, покажите сотруднику точки штрихкод из приложения или личного кабинета'
        assert case['additional_evidence_quote'] in docs[t['target_ids'][0]]['doc']
    cases.append(case)
suite={'schema_version':1,'corpus_sha256':digest(ROOT/'data/documents.jsonl'),
       'frozen_queries_sha256':hashes['queries.jsonl'],'frozen_tasks_sha256':hashes['tasks.jsonl'],
       'protocol':{'unit':'Five pre-registered known-item requests and two verified-empty requests in the VK research pilot, identical text and as-of scope.',
                   'reviewer':'Agent review against the complete original for each cited message; no human assessment.',
                   'automatic':'Fact patterns check evidence on one cited line. For a complete answer the source, entity, requested relation and all material assertions must also match on full-text review.',
                   'adjudication':{'complete_supported_answer':'Every requested fact is answered, supported by the cited original, and no added assertion contradicts or invents grounds.','partially_supported_answer':'At least one requested fact is supported but another is missing, or a citation independently supports only part of the response.','incorrect_factual_answer':'The answer asserts an incorrect or unsupported task fact. Pieces from other events cannot be combined as one.','irrelevant_excerpt':'The reply supplies an excerpt or data unrelated to the requested fact and does not resolve the question.','justified_abstention':'The requested evidence is absent and the reply explicitly says it could not be found or established.','unjustified_abstention':'The reply declines even though the registered source contains the requested evidence.'},
                   'currency':'For vk-pilot-01 only the number is stated in the original. Adding a currency is an unsupported claim.'},
       'cases':cases}
assert len(cases)==7 and sum(bool(c['facts']) for c in cases)==5
dest=FIX/'rag_facts.json'
assert not dest.exists()
dest.write_text(json.dumps(suite,ensure_ascii=False,indent=2)+'\n')
import re
for c in cases:
    for id in c['sources']:
        assert all(re.search(p,docs[id]['doc'],re.I) for p in c['facts']), c['id']
print(digest(dest),len(cases))
