"""Private local fixture builder. Generates aggregate-only output; never calls the service."""
from collections import Counter
from datetime import date, timedelta
from hashlib import sha256
import json
from pathlib import Path
import random
import re


root = Path(__file__).resolve().parent
raw = (root / 'corpus/documents.jsonl').read_bytes()
rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
records = {row['doc_id']: row for row in rows}
dates = {key: date.fromisoformat(row['doc_dt']) for key, row in records.items()}
texts = {key: row['doc'].casefold() for key, row in records.items()}
envelopes = {key: row['doc'].split('\n', 2) for key, row in records.items()}
subjects = {key: (lines[1].partition(':')[2].strip() if len(lines)>1 and re.match(r'^(?:тема|subject)\s*:',lines[1],re.I) else '') for key, lines in envelopes.items()}
bodies = {key: lines[2] if len(lines)>2 else '' for key,lines in envelopes.items()}
boxes = sorted({r['synthetic_user'] for r in rows})
box_for = {key: row['synthetic_user'] for key,row in records.items()}
assert len(boxes)==2
last = max(dates.values())
rng = random.Random(81473051)
positive, negative, checks = [], [], []
chosen = set()
word = r'(?:[A-Za-zА-Яа-яЁё]{3,21})'
phrase_re = re.compile(r'(?<![\w./@-])(?:' + word + r' ){3}' + word + r'(?![\w./@-])')
junk = re.compile(r'https?|image|class|style|background|padding|display|border|align|cellpadding|font.size|color|unsubscribe|subscribed|отписаться|отписки|cookie|парол|password|secret|token|username|логин|verification|однораз|login|auth|код подтвержд|access key',re.I)
months = ('','января','февраля','марта','апреля','мая','июня','июля','августа','сентября','октября','ноября','декабря')


def rus(day):
    return f'{day.day} {months[day.month]} {day.year} года'


def make(key, family, at, query, expected, patterns, window=None, box=None):
    mailbox = box or box_for[next(iter(expected))]
    out = dict(id=key, family=family, mailbox=mailbox, date=at.isoformat(), query=query,
               expected=sorted(expected), evidence_all=patterns)
    if window:
        out['received'] = [day.isoformat() for day in window]
    return out


def pick_phrases(family, count, source, min_offset=0, max_offset=None):
    ids = list(records)
    rng.shuffle(ids)
    picked=[]
    for key in ids:
        if key in chosen:
            continue
        text = source[key][min_offset:max_offset]
        if not text or (min_offset and len(text) < 32):
            continue
        candidates=list(phrase_re.finditer(text))
        rng.shuffle(candidates)
        for found in candidates[:12]:
            value = found.group()
            words = value.split()
            if len(set(w.casefold() for w in words)) < 4 or len(value) < 23 or junk.search(value):
                continue
            if len({char for char in value if char.islower()}) < 7:
                continue
            normalized=value.casefold()
            if sum(normalized in blob for blob in texts.values()) != 1:
                continue
            chosen.add(key)
            picked.append((key,value))
            break
        if len(picked)==count:
            return picked
    raise RuntimeError(f'Could not fill requested size of {family}.')


title_phrases = pick_phrases('subject', 10, subjects)
early_phrases = pick_phrases('body', 8, bodies, max_offset=900)
deep_phrases = pick_phrases('body-late', 6, bodies, min_offset=1900, max_offset=8500)
for family,pairs,field in [('exact_subject',title_phrases,'теме'),('exact_body',early_phrases,'тексте'),
                           ('exact_body_after_1900',deep_phrases,'тексте')]:
    for i,(key,phrase) in enumerate(pairs,1):
        question=f'Найди письмо, в {field} которого есть фраза «{phrase}».'
        positive.append(make(f'{family}-{i}',family,last,question,[key],[re.escape(phrase)]))

dated_phrases = pick_phrases('dated', 8, subjects)
for i,(key,phrase) in enumerate(dated_phrases):
    day = dates[key]
    if i%2:
        at = day + timedelta(days=1)
        question=f'Найди полученное вчера письмо со словами «{phrase}».'
        window = (day, day)
    else:
        start=day-timedelta(days=2)
        end=day+timedelta(days=1)
        at=end+timedelta(days=5)
        question=f'Ищу письмо со словами «{phrase}», полученное с {rus(start)} по {rus(end)}.'
        window=(start,end)
    positive.append(make(f'dates-{i+1}','exact_with_received_period',at,question,[key],[re.escape(phrase)],window))

# These natural-language descriptions are authored separately from the source;
# only generic explicit envelope signals are used to assign *silver* positives.
# The diagnostic makes no negative judgements about unlabelled messages.
topics=[
  ('receipt',r'кассовый чек|электронный чек|receipt for (?:your )?payment|your (?:payment |purchase )?receipt',
   'Где мне присылали подтверждающий документ после того, как я расплатился за покупку?'),
  ('calendar',r'invitation\s*:|приглашение на (?:встречу|событие)|invitation to (?:a |the )?(?:meeting|event)',
   'Кто присылал предложение добавить совместное мероприятие в мой календарь?'),
  ('deadline',r'дедлайн|deadline|due date|крайний срок',
   'Ищу сообщения, где говорят, к какому моменту нужно закончить дело.'),
  ('unread',r'непрочитанн.{0,18}сообщени|unread messages?',
   'Где напоминали, что в переписке остались новые реплики, которые я ещё не открыл?'),
  ('invite',r'invited you to|приглаша[ею]т вас (?:присоединиться|в проект|в команду)',
   'Где организаторы предлагают мне присоединиться к совместной работе?'),
  ('reminder',r'напоминание|напоминаем|reminder',
   'Найди письма, в которых меня ещё раз уведомляют о ранее запланированном деле.'),
]
for i,(label, pattern, prompt) in enumerate(topics):
    header = r'\A(?:От|Отправитель|From):[^\n]*\n(?:Тема|Subject):[^\n]*(?:'+pattern+')'
    regex = re.compile(header, re.I)
    target_ids=[key for key,row in records.items() if regex.search(row['doc'])]
    rng.shuffle(target_ids)
    # Two distinct date windows with at least five unlabelled distractors; the
    # target status signal is independently checked in the actual subject line.
    used=set()
    for size in (4,10):
        for anchor in target_ids:
            mid=dates[anchor]; lo=mid-timedelta(days=size//2); hi=lo+timedelta(days=size-1)
            relevant=[key for key in target_ids if box_for[key]==box_for[anchor] and lo<=dates[key]<=hi]
            eligible=sum(lo<=d<=hi for key,d in dates.items() if box_for[key]==box_for[anchor])
            if not (1 <= len(relevant) <= 4) or eligible-len(relevant) < 5 or used.intersection(relevant):
                continue
            used.update(relevant)
            q=prompt+f' Интересуют только письма, полученные с {rus(lo)} по {rus(hi)}.'
            positive.append(make(f'semantic-{i+1}-{size}','authored_query_silver_subject',hi+timedelta(days=6),q,relevant,[header],(lo,hi)))
            break
        else:
            # A rare class may not have two distinct populated windows.
            continue

for i,(key,phrase) in enumerate(title_phrases[:4]):
    at=dates[key]-timedelta(days=1)
    negative.append(make(f'future-{i+1}','unique_phrase_before_receipt',at,
                         f'Где письмо с точными словами «{phrase}»?',[],[re.escape(phrase)],box=box_for[key]))
    other=next(box for box in boxes if box!=box_for[key])
    negative.append(make(f'other-{i+1}','unique_phrase_in_other_mailbox',last,f'Найди письмо, где буквально встречается «{phrase}».',[],[re.escape(phrase)],box=other))
for i,marker in enumerate(['SYN-9cd19e26341b','ZXW-740392518662','АльтаметрияСемиДюн','КалейдоскопПолюсаТридцать']):
    assert not any(marker.casefold() in blob for blob in texts.values())
    negative.append(make(f'absent-{i+1}','absent_literal',last,
                         f'Покажи письма, в которых буквально встречается «{marker}».',[],[re.escape(marker)],box=boxes[i%len(boxes)]))

for i,(key,_) in enumerate(title_phrases[:3]):
    day=dates[key]
    mailbox=box_for[key]
    other=next(box for box in boxes if box!=mailbox)
    for tag,box,when,status in [('day',mailbox,day,200),('before',mailbox,day-timedelta(days=1),404),('foreign',other,last,404)]:
        checks.append(dict(id=f'read-{i+1}-{tag}',mailbox=box,date=when.isoformat(),doc_id=key,status=status))

suite=dict(name='Локальная диагностика импорта почты',version=1,
           documents_sha256=sha256(raw).hexdigest(),
           method=[
               'Фрагменты и все метки зафиксированы автоматически до первого запроса к приложению. Для буквальных фрагментов и отрицательных маркеров проверена уникальность/отсутствие в полном корпусе.',
               'У естественных вопросов формулировки составлены отдельно вручную, но сами связи с письмами вычислены по явным признакам темы. Другие релевантные документы возможны. Этим связям соответствует слабая разметка; ручной поштучной разметки или официальных qrels нет.',
               'Периоды и видимость проверяются по исходной календарной дате doc_dt, сохранённой без преобразования. Для датированных естественных запросов выбраны интервалы с одним–четырьмя известными примерами и минимум пятью другими письмами.',
           ],
           global_anchors=[],cases=positive+negative,read_checks=checks)
out=root/'probes.json'
encoded=(json.dumps(suite,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
if out.exists() and out.read_bytes()!=encoded:
    raise RuntimeError('Previously frozen fixture differs. Preserve the earlier run.')
out.write_bytes(encoded)
print('families:',dict(Counter(c['family'] for c in suite['cases'])))
print('search checks:',len(suite['cases']),'read checks:',len(checks),'fixture SHA-256:',sha256(encoded).hexdigest())
