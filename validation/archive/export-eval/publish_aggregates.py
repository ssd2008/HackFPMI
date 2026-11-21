"""Only public category labels and numeric counts enter the shareable report."""
from pathlib import Path
import json
import hashlib
sources = {key: hashlib.sha256((Path('mailsearch') / (key + '.py')).read_bytes()).hexdigest()
           for key in ['retrieval', 'dates', 'spelling', 'dense', 'app']}
assert sources['retrieval'] == '93e475250605d17ba1b8740d7d552eaea4d1340a999c3349d8ea2a695141599f'
report = {
    'schema': 1, 'source_sha256': sources,
    'protocol': {
        'evaluation': 'Самостоятельные диагностические наборы. Для этих архивов нет внешней разметки релевантности.',
        'positive_technical': 'Искомые фрагменты темы и тела встречаются ровно в одном письме всего соответствующего корпуса. Часть запросов также ограничивает даты получения.',
        'positive_natural_language': 'Формулировки составлены вручную без просмотра выдачи. Релевантные примеры выбраны автоматически по явным признакам темы. Ручной оценки писем не проводилось; другие релевантные письма возможны.',
        'ranking': 'Для положительных запросов учитывается попадание хотя бы одного подтверждённого письма в первые 1 и 5.',
        'negative': 'Строго пустая выдача требуется для отсутствующего буквального маркера либо для уникального фрагмента из будущего или другого ящика.',
        'scope': 'Независимая проверка каждого возвращённого ID по корпусу, ящику, дате запроса и явно указанному периоду. Всегда запрашивалось не более 10 писем.',
        'timing': 'Один последовательный проход по зафиксированным запросам каждого архива после запуска сервера. Миллисекунды из Server-Timing, ближайший ранг. Замеры по архивам выполнялись по очереди.',
    },
    'corpora': [],
}
specs=[('raw_export','export-eval',5939,1,263,'UTC timestamps to Europe/Moscow calendar dates'),
       ('native_export','export-eval-02',3427,2,0,'supplied YYYY-MM-DD, original partitions and IDs retained')]
for corpus,folder,count,mailboxes,shifted,date_policy in specs:
    f=Path('.cache')/folder; s=json.loads((f/'final-pass-summary.json').read_text()); metrics=s['metrics']; gs=s['groups']
    def positives(rows):
        return {new:sum(row[old] for row in rows) for new,old in [('queries','positive_queries'),('hit_at_1','hit_at_1'),('hit_at_5','hit_at_5')]}
    def empties(rows):
        return {'queries':sum(row['empty_queries'] for row in rows), 'correctly_empty':sum(row['correctly_empty'] for row in rows)}
    technical=[v for k,v in gs.items() if k.startswith('exact_')]
    semantic=gs['authored_query_silver_subject']
    obj={
        'corpus':corpus,'input_messages':count,'mailboxes':mailboxes,'missing_dates':0,
        'original_date_changed_by_timezone':shifted,'calendar_day_policy':date_policy,
        'frozen_probe_file_sha256':s['fixture_sha256'],
        'search_requests':s['distinct_sequential_requests'],
        'technical':positives(technical), 'natural_language_silver':positives([semantic]),
        'negative':empties(gs.values()),
        'positive_by_probe_family':{k:positives([v]) for k,v in gs.items() if k.startswith('exact_')},
        'negative_by_probe_family':{k:empties([v]) for k,v in gs.items() if v['empty_queries']},
        'scope':{k:metrics[k] for k in ['returned_items','invalid_mailbox_or_asof_items','outside_explicit_received_window_items']},
        'direct_reads':{'checks':metrics['read_checks'],'passed':metrics['correct_reads']},
        'inbox_smoke':{'mailboxes_opened':mailboxes,'sorted_pages_checked':True,'original_message_checked':True,
                       'empty_snapshot_checked':True,'nonexistent_document_returns_404':True,'rag_without_evidence_has_no_sources':True},
        'sequential_server_ms':s['latency']
    }
    report['corpora'].append(obj)
target=Path('results/additional_corpora.json')
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('Public aggregate report written for',len(report['corpora']),'archives.')
