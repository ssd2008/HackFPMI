"""Freeze rule-adjudicated VK tasks directly from the public corpus before either run."""
import collections
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEST = ROOT/'tests/fixtures/validation_pilot'
boxes = ['box-12259a301ee2','box-fad2b057e79a','box-fda57474896e','box-90560850fd8e','box-a55c522c6949']
samples = [
 # Split, intent, source mailbox, as-of date, query, grade 2, grade 1, family/chain.
 ('dev','known_item',0,'2028-08-21','Какую сумму в 2027 году выставили Фадею Гурьеву за наём муниципального жилья?',
  'Счёт 2027 года Гурьеву за муниципальный наём с указанной суммой.','Сообщение 2027 года о том же счёте Гурьева без его суммы.','municipal-hire-2027-invoice'),
 ('dev','known_item',1,'2028-02-15','Найди заказ на ударный гайковёрт с пневмоприводом и моментом 780 Нм. Сколько он стоил?',
  'Заказ самого ударного пневмогайковёрта на 780 Нм с ценой.','То же изделие или номер заказа без цены. Насадки и запчасти для других заказов получают 0.','air-impact-wrench-28626-1707-1'),
 ('dev','known_item',1,'2028-01-15','Как получить возвраты и сколько их всего ждали в уведомлении, где отдельно числились три товара по схеме FBO?',
  'Памятка по получению возвратов с тремя товарами FBO и общим числом ожидающих возвратов.','Памятка того же продавца о получении и хранении возвратов в пункте без этого конкретного количества FBO.','pickup-three-fbo'),
 ('dev','collection',0,'2028-08-22','Собери письма о пенях за муниципальный наём, полученные 20 августа 2028 года. В первую очередь интересуют начисления на имя Фадея Гурьева.',
  'В указанный день получен штраф за пени и Гурьев прямо назван адресатом начисления.','Другие пришедшие в тот же день штрафы за пени по муниципальному найму.','rent-penalties-received-2028-08-20'),
 ('dev','collection',1,'2028-01-15','Покажи заказы с метчиками для нарезания резьбы. Особенно нужны наборы ручных метчиков по металлу.',
  'Заказ на набор ручных метчиков по металлу.','Другие заказы, в состав которых входит метчик, в том числе набор восстановления свечной резьбы.','thread-cutting-taps'),
 ('dev','empty',1,'2028-01-15','Есть ли уведомление об отмене заказа SYN-5d701b449b?',
  'Уведомление об отмене именно этого заказа.','Сообщение с этим же номером, где отмена не подтверждена.','absent-marketplace-order-5d701b449b'),
 ('blind','known_item',4,'2028-03-08','Какой код присвоен иммуногенетике в письме с перечнем исследований крови и микробиологии?',
  'Перечень исследований с отдельным кодом иммуногенетики и разделом микробиологии.','Документ о том же сообщении или отдельная строка об иммуногенетике без ответа. Служебные коды качества без перечня исследований получают 0.','laboratory-examinations-f4be7bca7e'),
 ('blind','known_item',2,'2028-08-15','Кто работал с этапом производства ОНПР-53.7.1, когда программа сообщила, что у объекта нет поля?',
  'Ошибка отсутствующего поля для этапа ОНПР-53.7.1 с указанным пользователем.','Другие упоминания того же этапа без полного ответа. Общее имя сервера без этой операции получает 0.','production-stage-onpr-53-7-1'),
 ('blind','collection',3,'2028-07-01','Собери заказы на средства для замёрзших автомобильных замков. В приоритете те, которыми размораживают и замок, и стекло.',
  'Заказ на средство для размораживания и замков, и стекла.','Средство от наледи, рекомендованное для замков без одновременного упоминания стекла. Скребки только для стёкол получают 0.','locks-and-glass-deicer'),
 ('blind','collection',2,'2028-07-14','Собери пришедшие с 11 по 12 июля 2028 года отчёты о недоступности с фамилией Тетерина. В первую очередь те, где вместе с ней указана Артемьева.',
  'В окно получения обе фамилии присутствуют в сообщении о недоступности.','В окно получения в сообщении о недоступности названа Тетерина с другим участником. Имя оператора в реквизитах отдельно не засчитывается.','availability-2028-07-11-to-12'),
 ('blind','collection',4,'2028-03-05','Найди письма в медико-санитарную часть №157, полученные с 1 по 3 марта 2028 года. Сначала покажи те, где код внешнего контроля ФСВОК равен 15226.',
  'Письмо адресовано МСЧ №157 в указанный период и в поле ФСВОК стоит 15226.','Иное письмо этому же получателю в тот же период. Номер лицензии и международный код EQAS отдельно не подтверждают запрошенный код.','medical-unit-157-march-fsvok'),
 ('blind','empty',4,'2028-03-08','Найди пришедшее 5 марта 2028 года письмо с посадочным QR-кодом на паром.',
  'Письмо о посадке на паром с QR-кодом, полученное в этот день.','Подтверждение той же переправы без самого кода. Электронные лабораторные коды получают 0.','absent-ferry-boarding-march-5'),
]
targets={0:'L-d2db3bf9b0e8873699f7',1:'L-9822b31c1104f4365ef7',2:'L-6a424bc0727869055ae7',6:'L-bd1cb09f05969c6fdd9b',7:'L-007247dd284d021f4391'}
facts={0:('504.75, валюта в письме не указана','Счёт от 20.08.2027 на 504.75.'),
       1:('6655.17 RUB','780 Нм. Сумма заказа — 6655.17 RUB.'),
       2:('457 возвратов. Для получения показать штрихкод.','Всего в точке выдачи ждут 457 возвратов.'),
       6:('416','иммуногенетика — 416'),7:('Евдокимова Вероника Харитоновна','Пользователь: Евдокимова Вероника Харитоновна.')}
windows={3:('2028-08-20','2028-08-20'),9:('2028-07-11','2028-07-12'),10:('2028-03-01','2028-03-03'),11:('2028-03-05','2028-03-05')}

def grade(i,r):
 s=r['doc'].casefold().replace('ё','е')
 if i in windows:
  a,b=windows[i]
  if r['doc_dt'] is None or not a<=r['doc_dt']<=b: return 0
 if i==0:
  if r['doc_dt'] and r['doc_dt'][:4]=='2027' and 'гурьев' in s and 'по найму' in s:
   return 2 if '504.75' in s and 'счет' in s else 1
 elif i==1:
  if r['doc_id']==targets[1]:return 2
  if '28626-1707-1' in s:return 1
 elif i==2:
  if 'возврат' in s and 'точке выдачи' in s and '10 дней' in s and 'штрихкод' in s:
   return 2 if re.search(r'\b3 товара по схеме fbo',s) and '457 возвратов' in s else 1
 elif i==3:
  if 'штраф' in s and 'пени' in s and ('найм' in s or 'наем' in s or 'найму' in s):
   return 2 if re.search(r'гурьев\w*\s+фаде\w*',s.split('по документу')[0].split('по данным документа')[0]) else 1
 elif i==4:
  if 'метчик' in s:
   return 2 if re.search(r'ручн\w*\s+метчик',s) and 'металл' in s else 1
 elif i==5:
  if 'syn-5d701b449b' in s:return 2 if 'отмен' in s else 1
 elif i==6:
  if 'иммуногенет' in s or 'syn-f4be7bca7e' in s:
   return 2 if re.search(r'иммуногенетика\s*[-—:]\s*416',s) and 'микробиолог' in s else 1
 elif i==7:
  if 'онпр-53.7.1' in s:
   return 2 if 'поле объекта не обнаружено' in s and 'евдокимова вероника' in s else 1
 elif i==8:
  if re.search(r'зам(?:ок|к[аиоу])',s) and re.search(r'размор|обледен|антилед',s):
   return 2 if re.search(r'стек[ло]',s) else 1
 elif i==9:
  # Look in the assertion preceding the operator field, not the session metadata.
  assertion=s.split('пользователь')[0]
  if 'недоступ' in assertion and 'тетери' in assertion:
   return 2 if 'артемьев' in assertion else 1
 elif i==10:
  if re.search(r'медико.санитарная часть.{0,18}157',s):
   return 2 if re.search(r'фсвок[^\d]{0,12}15226\b',s) else 1
 elif i==11:
  if re.search(r'паром|переправ',s):return 2 if re.search(r'qr|посадочн',s) else 1
 return 0

def rows(path):return [json.loads(l) for l in path.read_text().splitlines()]
def write(name,values):
 raw=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in values).encode()
 p=DEST/name
 if p.exists():assert p.read_bytes()==raw,p
 else:p.write_bytes(raw)
 return hashlib.sha256(raw).hexdigest()

docs=rows(ROOT/'data/documents.jsonl'); dby={d['doc_id']:d for d in docs}
old={k for f in ['search_probes','search_probes_extra'] for c in json.loads((ROOT/f'tests/fixtures/{f}.json').read_text())['cases'] for k in c['expected']}
official_boxes={r['synthetic_user'] for r in rows(ROOT/'data/queries_dev.jsonl')}
queries=[];tasks=[];qrels=[]; positive_ids=set(); seen_titles=set()
for i,(split,typ,b,asof,query,direct,context,family) in enumerate(samples):
 qid='vk-pilot-%02d'%(i+1); box=boxes[b];assert box not in official_boxes
 queries.append({'query_id':qid,'synthetic_user':box,'query':query,'query_date':asof})
 visible=[r for r in docs if r['synthetic_user']==box and (r['doc_dt'] is None or r['doc_dt']<=asof)]
 selected={r['doc_id']:grade(i,r) for r in visible}; hit={k for k,v in selected.items() if v>0}
 assert not hit&old and not hit&positive_ids,(i,old&hit,positive_ids&hit)
 positive_ids |= hit
 grades={g:[key for key,value in selected.items() if value==g] for g in (2,1)}
 if typ=='known_item':assert grades[2]==[targets[i]],(i,grades)
 elif typ=='collection':assert len(hit)>=2 and grades[2],(i,grades)
 else:assert not hit,(i,hit)
 t={'query_id':qid,'family_id':family,'chain_id':box+'/'+family,'split':split,'source':'vk',
    'owner_id':box,'synthetic_user':box,'query_date':asof,'scenario':family,'state':'verified_empty' if not hit else 'positive',
    'task_type':typ,'target_ids':[targets[i]] if i in targets else [],'judgment':'agent_rules_exhaustive',
    'grade_2_criterion':direct,'grade_1_criterion':context,'grade_0_criterion':'Остальные доступные письма, в том числе другие события и письма вне явно указанного периода.',
    'grade_2_ids':grades[2],'grade_1_ids':grades[1],'visible_count':len(visible)}
 if i in facts:
  val,fragment=facts[i];assert fragment in dby[targets[i]]['doc'],(i,fragment)
  t['required_fact']=val;t['evidence_quote']=fragment
 if i==2:
  t['additional_evidence_quote']='штрихкод'
  assert t['additional_evidence_quote'] in dby[targets[i]]['doc'].lower()
 if i in windows:
  t['received_window']=list(windows[i]);lo,hi=windows[i];t['in_window_count']=sum(lo<=(r['doc_dt'] or '0')<=hi for r in visible)
 if i==5:t['empty_proof']='Точный идентификатор SYN-5d701b449b отсутствует во всех 16 376 текстах, проверено без учёта регистра.'
 if i==11:t['empty_proof']='Полный срез ящика за 05.03.2028 проверен на паром, переправу, посадочные документы и QR-коды. В корпусе отсутствуют упоминания парома.'
 assert family not in seen_titles;seen_titles.add(family);tasks.append(t)
 for did,value in selected.items():qrels.append({'query_id':qid,'doc_id':did,'relevance':value})
 print(qid,split,typ,len(visible),len(grades[2]),len(grades[1]),t.get('in_window_count',''))
old_texts={hashlib.sha256(' '.join(dby[k]['doc'].casefold().split()).encode()).hexdigest() for k in old}
assert not {hashlib.sha256(' '.join(dby[k]['doc'].casefold().split()).encode()).hexdigest() for k in positive_ids}&old_texts
assert all('syn-5d701b449b' not in x['doc'].casefold() and 'паром' not in x['doc'].casefold() for x in docs)
assert set(t['owner_id'] for t in tasks if t['split']=='dev').isdisjoint(t['owner_id'] for t in tasks if t['split']=='blind')
hashes={name:write(name,vs) for name,vs in [('queries.jsonl',queries),('tasks.jsonl',tasks),('qrels.jsonl',qrels)]}
freeze=hashlib.sha256('\n'.join(name+' '+hashes[name] for name in sorted(hashes)).encode()).hexdigest()
print(len(qrels),'qrels,',len(positive_ids),'distinct positive letters')
print(json.dumps({'files':hashes,'freeze':freeze},indent=2)); (HERE/'freeze.json').write_text(json.dumps({'files':hashes,'freeze':freeze},indent=2)+'\n')
