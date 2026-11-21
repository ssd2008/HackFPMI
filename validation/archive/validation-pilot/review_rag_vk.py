"""Verify the two unchanged RAG reports against the original mailbox texts and record agent adjudications."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
PILOT=ROOT/'results/validation_pilot'
FIX=ROOT/'tests/fixtures/validation_pilot/rag_facts.json'
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path): return json.loads(path.read_text())
suite=load(FIX)
docs={r['doc_id']:r for r in map(json.loads,(ROOT/'data/documents.jsonl').read_text().splitlines())}
reports={v:load(PILOT/('rag-'+v+'.json')) for v in ('before','after')}
ranks={v:load(PILOT/(v+'.json'))['queries'] for v in reports}
seen={v:{r['id']:r for r in rep['cases']} for v,rep in reports.items()}
assert all(rep['fixture_sha256']==digest(FIX) for rep in reports.values())
assert all(rep['corpus_sha256']==suite['corpus_sha256'] for rep in reports.values())
# Per-line review of the material, after reading the complete cited originals.
notes={
 'L-6a424bc0727869055ae7':('context_for_requested_event','288 FBS и 3 FBO действительно указаны в письме. Суммарные 457 и требование показать штрихкод в ответе отсутствуют.'),
 'L-1d757a7af1a30d3053c2':('other_event','136 FBS относятся к другому уведомлению. Всего в нём 747 возвратов. Оно не подтверждает число в запрошенном уведомлении.'),
 'L-2b36e9dec87befbfe977':('other_event','7170.37 RUB — сумма заказа на сцепление, к выдаче возвратов не относится.'),
 'L-41060362bca6f66c6707':('other_event','Идентификатор и сумма 3047.79 RUB соответствуют отдельному заказу, к выдаче возвратов не относятся.'),
 'L-bd1cb09f05969c6fdd9b':('answers_requested_fact','В перечне иммуногенетике непосредственно приписан код 416. Другие исследования с таким же номером не меняют этой связи.'),
 'L-007247dd284d021f4391':('context_for_requested_event','Этап и время названы точно. Имя пользователя есть в оригинале, но отсутствует в ответе.'),
 'L-90f99c21faf719433087':('context_for_requested_event','Этап, дата и топоним совпадают с оригиналом. Он описывает отдельную ошибку на том же этапе и называет другого пользователя, но ответ не называет его.'),
 'L-68d750048f4200051628':('other_event','Этап ОНПР-584.5.2 и его дата реальны, но это не запрошенный ОНПР-53.7.1. Пользователя реплика не называет.'),
 'L-bb48b61ec7dacb193b6b':('other_event','Тема и дата принадлежат письму о внешнем контроле качества. Посадки, парома и QR-кода в нём нет.'),
 'L-3697244be91b0ea6e091':('other_event','Данные темы принадлежат ещё одному письму о внешнем контроле качества. Они не подтверждают запись на паром.'),
 'L-cac9ceb839bc02db1d4d':('other_event','Дата и получатель принадлежат письму в МСЧ №157 о контроле качества, которое не подтверждает посадку на паром.')
}
cases_review={
 'vk-pilot-01':('unjustified_abstention','Источник прямо содержит счёт на 504.75, без указания валюты. Обе версии отказали.'),
 'vk-pilot-02':('unjustified_abstention','Письмо о пневмогайковёрте 780 Нм прямо указывает сумму 6655.17 RUB. Обе версии отказали.'),
 'vk-pilot-03':('irrelevant_excerpt','Обе версии дали контекстные и посторонние выдержки. Ни общий итог 457, ни предъявление штрихкода в ответ не попали. Числа 288 и 3 описывают отдельные категории, 136 — отдельное уведомление.'),
 'vk-pilot-06':('justified_abstention','Идентификатора заказа SYN-5d701b449b нет в корпусе. Обе версии ответили, что подтверждения недостаточно.'),
 'vk-pilot-07':('complete_supported_answer','Ответ 416 дан с номером источника. В оригинале именно иммуногенетике явно приписан этот код.'),
 'vk-pilot-08':('irrelevant_excerpt','Обе версии дали описания этапов и дат, а пользователь не назван ни в одной выдержке. Два исходных подходящих письма содержат разные имена.'),
 'vk-pilot-12':('irrelevant_excerpt','Вместо отказа приведены реквизиты писем о контроле качества за нужную дату. Ни текста посадочного QR-кода, ни утверждения, что посадка подтверждена, ответы не содержат.')
}
review=[]
for task in suite['cases']:
    qid=task['id']
    category,decision=cases_review[qid]
    item={'query_id':qid,'split':task['split'],'expected_outcome':task['expected_outcome'],
          'expected':task['expected'],'assessment':decision,'versions':{}}
    for v in reports:
        case=seen[v][qid]
        evidence=case['opened_source_ids']
        lines=case['answer'].splitlines()
        citations=[]
        for line in lines:
            refs=[int(n) for n in re.findall(r'\[(\d+)\]',line)]
            for n in refs:
                did=evidence[n-1]
                d=docs[did]
                assert d['synthetic_user']==task['mailbox'] and (d['doc_dt'] is None or d['doc_dt']<=task['date'])
                if qid=='vk-pilot-12': assert d['doc_dt']=='2028-03-05'
                if case['mode']=='extractive':
                    match=re.fullmatch(r'«(.*)»\s+\[\d+\]\.',line)
                    assert match,line
                    fragment=match[1]
                    assert fragment in d['doc'],(qid,did,fragment)
                    support='verbatim_in_cited_original'
                elif case['mode']=='generative':
                    assert qid=='vk-pilot-07' and line=='416 [1].'
                    fragment='иммуногенетика — 416'
                    assert fragment in d['doc']
                    support='generated_claim_supported_by_cited_original'
                else: raise AssertionError(case['mode'])
                role,comment=notes[did]
                citations.append({'citation':n,'source':did,'support':support,'source_excerpt':fragment,
                                  'role_for_question':role,'review':comment})
        assert len(citations)==case['citations']==len(evidence)
        if case['mode']=='no_evidence':assert case['answer']=='В найденных письмах недостаточно сведений для ответа.'
        target=task['sources']
        ranking=ranks[v][qid]['ranked_ids']
        item['versions'][v]={
            'class':category,'mode':case['mode'],'requested_fact_claimed_in_error':False,
            'required_facts_supported':case['facts_cited'],'required_facts_total':case['facts_expected'],
            'target_rank_of_saved_search':min((ranking.index(t)+1 for t in target if t in ranking),default=None),
            'citations':citations
        }
    review.append(item)
summary={}
for v in reports:
    counts=Counter(item['versions'][v]['class'] for item in review)
    def refs(items): return [x for item in items for x in item['versions'][v]['citations']]
    pos=[i for i in review if i['expected_outcome']=='supported_answer']
    neg=[i for i in review if i['expected_outcome']=='justified_abstention']
    all_refs=refs(review); pos_refs=refs(pos)
    summary[v]={
        'answerable':len(pos),'complete_supported_answers':counts['complete_supported_answer'],
        'unjustified_abstentions':counts['unjustified_abstention'],'answerable_nonsolving_excerpts':sum(i['versions'][v]['class']=='irrelevant_excerpt' for i in pos),
        'answerable_partially_supported':counts['partially_supported_answer'],'incorrect_requested_fact_asserted':counts['incorrect_factual_answer'],
        'empty':len(neg),'justified_abstentions':counts['justified_abstention'],
        'empty_nonsolving_excerpts':sum(i['versions'][v]['class']=='irrelevant_excerpt' for i in neg),
        'empty_false_requested_facts_asserted':sum(i['versions'][v]['requested_fact_claimed_in_error'] for i in neg),
        'total_cited_lines':len(all_refs),'cited_lines_supported_by_own_letter':len(all_refs),
        'positive_cited_lines':len(pos_refs),'positive_cited_lines_delivering_a_required_fact':sum(x['role_for_question']=='answers_requested_fact' for x in pos_refs),
        'positive_expected_source_citations':reports[v]['summary']['expected_source_citations_on_answerable'],
        'required_fact_units_cited':reports[v]['summary']['required_facts_with_citation'],'required_fact_units':reports[v]['summary']['required_facts'],
        'response_modes':reports[v]['summary']['modes']
    }
related=['L-007247dd284d021f4391','L-90f99c21faf719433087']
candidates=[('поле объекта не обнаружено','ОНПР-53.7.1','Пользователь: Евдокимова Вероника Харитоновна.'),('Поле объекта не обнаружено.','ОНПР-53.7.1','Пользователь: Анатолий Захарьевич Шаров.')]
ambiguity=[]
for did,quotes in zip(related,candidates):
    assert all(q in docs[did]['doc'] for q in quotes)
    ambiguity.append({'doc_id':did,'fragments':quotes})
result={
    'protocol':'Эндпоинт /api/answer, исходный текст, ящик и дата семи зарегистрированных запросов. Один последовательный проход каждой версии. Автоматическая проверка открыла каждую ссылку в том же ящике на ту же дату и сравнила полный текст с корпусом.',
    'assessment':'Агент сверил все фактические строки ответов с полными оригиналами по каждой ссылке. Дословная выдержка, описывающая другую операцию, подтверждается своим письмом, но не решает задачу. Независимая человеческая оценка отсутствует.',
    'measurement_note':'Подсчёт заявленных ошибочных фактов по пустому запросу отделён от неуместной выдачи выдержек. Обе версии не придумали реквизиты посадки, однако выдали постороннюю корреспонденцию вместо отказа.',
    'version':{'before':'5fe3f5d','after_ranker':'d747e72d'},
    'sha256':{'corpus':suite['corpus_sha256'],'fixture':digest(FIX),**{v:digest(PILOT/('rag-'+v+'.json')) for v in reports}},
    'summary':summary,
    'annotation_ambiguity':{'query_id':'vk-pilot-08',
        'reason':'Исходный вопрос содержит код этапа и ошибку отсутствующего поля, но не дату самого сообщения, базу или другие признаки инцидента. Два разных подходящих сообщения называют разных пользователей. Зарегистрированная цель содержит только Евдокимову; второй документ ранее получил 1 балл.',
        'evidence':ambiguity,'observed_effect':'Обе версии приводят выдержки из обоих сообщений и ни одного имени. Семантический исход одинаков при учёте обеих допустимых фамилий. Для самостоятельной оценки поиска эта неоднозначность ограничивает интерпретацию меток.'},
    'cases':review
}
p=PILOT/'rag-review.json'
assert not p.exists()
p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False))
