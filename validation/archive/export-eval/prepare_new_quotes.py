"""Build fresh quote controls offline, excluding all previously confirmed target letters."""
from pathlib import Path
from functools import lru_cache
from datetime import date, timedelta
from collections import Counter
from hashlib import sha256
import json
import random
import re
import sys

folder = Path(sys.argv[1])
raw = (folder/'corpus/documents.jsonl').read_bytes()
prior = json.loads((folder/'probes.json').read_text())
rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
by_id = {row['doc_id']: row for row in rows}
docs = {key: row['doc'].casefold() for key,row in by_id.items()}
boxes = {key: row['synthetic_user'] for key,row in by_id.items()}
days = {key: date.fromisoformat(row['doc_dt']) for key,row in by_id.items()}
latest = max(days.values())
old_ids = {key for case in prior['cases'] for key in case['expected']}
used = set(old_ids)
rng = random.Random(641972 if folder.name.endswith('02') else 316812)
cases = []
parts = {key: row['doc'].split('\n',2) for key,row in by_id.items()}
subjects = {key: (ls[1].partition(':')[2].strip() if len(ls)>1 and re.match(r'^(?:тема|subject)\s*:',ls[1],re.I) else '') for key,ls in parts.items()}
bodies = {key: ls[2] if len(ls)>2 else '' for key,ls in parts.items()}
word = r'[A-Za-zА-Яа-яЁё]{3,21}'
def words(length):
    return re.compile(r'(?<![\w./@-])(?:'+word+r' ){'+str(length-1)+r'}'+word+r'(?![\w./@-])')
phrase3, phrase4, phrase2 = words(3), words(4), words(2)
junk = re.compile(r'https?|image|class|style|background|padding|display|border|align|cellpadding|font.size|color|unsubscribe|subscribed|отписаться|отписки|cookie|парол|password|secret|token|username|логин|verification|однораз|login|auth|код подтвержд|access key',re.I)
@lru_cache(maxsize=768)
def included(text):
    folded=text.casefold()
    return frozenset(key for key,body in docs.items() if folded in body)

def pick(source, offset=0, limit=None, box=None):
    candidates = list(by_id); rng.shuffle(candidates)
    for key in candidates:
        if key in used or (box is not None and boxes[key] != box):
            continue
        part = source[key][offset:limit]
        matches = list(phrase4.finditer(part)) + list(phrase3.finditer(part))
        rng.shuffle(matches)
        for found in matches[:14]:
            phrase=found.group()
            if junk.search(phrase) or len(phrase)<21 or len(set(phrase.casefold().split())) != len(phrase.split()):
                continue
            if included(phrase)=={key}:
                used.add(key)
                return key, phrase
    raise RuntimeError('Unable to select sufficient disjoint unique phrases')

def quote(s,i):
    return '«'+s+'»' if i%2==0 else '"'+s+'"'
def add(family, query, box, day, literals, expected):
    possible=set.intersection(*(set(included(x)) for x in literals))
    actual={key for key in possible if boxes[key]==box and days[key]<=day}
    assert actual==set(expected)
    index=len(cases)+1
    cases.append(dict(id=f'q{index:02d}',family=family,mailbox=box,date=day.isoformat(),query=query,
                      expected=sorted(expected),evidence_all=[re.escape(x) for x in literals]))
for family,src,start,end,count in [('new_exact_subject',subjects,0,None,3),('new_exact_body',bodies,0,900,1),('new_exact_body_late',bodies,1900,9500,2)]:
    for i in range(count):
        key,fragment=pick(src,start,end)
        phrase=fragment.lower() if i%2 else fragment
        q='Открой письмо со словами '+quote(phrase,len(cases))+'.'
        add(family,q,boxes[key],latest,[phrase],[key])
for i in range(2):
    for _ in range(60):
        key,unique=pick(subjects)
        others=list(phrase2.finditer(bodies[key][:2100])) + list(phrase2.finditer(subjects[key]))
        rng.shuffle(others);common=None
        for m in others[:30]:
            value=m.group()
            if len(value)<10 or junk.search(value) or value.casefold() in unique.casefold():
                continue
            matched=included(value)
            if 2<=sum(boxes[k]==boxes[key] and days[k]<=latest for k in matched)<=250:
                common=value;break
        if common:break
    else:raise RuntimeError('No usable two-literal sample')
    q='Ищу письмо, где встречаются и '+quote(common,i)+', и '+quote(unique,i+1)+'.'
    add('new_two_literals',q,boxes[key],latest,[common,unique],[key])
for i in range(2):
    while True:
        key,original=pick(bodies,0,1800)
        words_=original.split()
        changed=' '.join([words_[1],words_[0],*words_[2:]])
        if not included(changed): break
    q='Найди точное словосочетание '+quote(changed,i)+'.'
    add('new_absent_word_order',q,boxes[key],latest,[changed],[])
for i in range(2):
    a,left=pick(subjects)
    b,right=pick(bodies,0,1600,box=boxes[a])
    assert a!=b and included(left).isdisjoint(included(right))
    q='Ищу одно письмо, в котором одновременно написаны '+quote(left,i)+' и '+quote(right,i+1)+'.'
    add('new_incompatible_literals',q,boxes[a],latest,[left,right],[])
while True:
    key,original=pick(subjects)
    pos=len(original)//2
    if not original[pos].isalpha():continue
    replacement='ц' if re.search('[А-Яа-яЁё]', original[pos]) else 'q'
    if original[pos].lower()==replacement:continue
    near=original[:pos]+replacement+original[pos+1:]
    if not included(near):break
add('new_one_letter_difference','Где буквально написано '+quote(near,1)+'?',boxes[key],latest,[near],[])
key,fragment=pick(subjects)
add('new_future_literal','Найди фразу '+quote(fragment,0)+'.',boxes[key],days[key]-timedelta(days=1),[fragment],[])
if len(set(boxes.values()))>1:
    key,fragment=pick(bodies,0,1100)
    other=next(k for k in sorted(set(boxes.values())) if k!=boxes[key])
    add('new_other_mailbox_literal','Найди точную фразу '+quote(fragment,1)+'.',other,latest,[fragment],[])
assert not old_ids.intersection({k for c in cases for k in c['expected']})
suite=dict(name='Новые проверки точной фразы',version=1,documents_sha256=sha256(raw).hexdigest(),
           global_anchors=[],cases=cases,read_checks=[],
           method=['Независимый детерминированный отбор по исходному корпусу. Все использованные письма исключены из положительных примеров предыдущего набора.',
                   'Положительная цель однозначна по всем буквальным фрагментам и доступности. Сопоставление без учёта регистра. Отрицательные примеры перепроверены полным проходом по корпусу.'])
b=(json.dumps(suite,ensure_ascii=False,indent=2)+'\n').encode()
out=folder/'fresh-quotes.json'
if out.exists() and out.read_bytes()!=b:raise RuntimeError('Frozen fixture differs; preserve earlier version')
out.write_bytes(b)
print('new positive',sum(bool(c['expected']) for c in cases),'new empty',sum(not c['expected'] for c in cases),
      'previous target overlap 0; SHA-256',sha256(b).hexdigest())
