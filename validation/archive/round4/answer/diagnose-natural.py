from pathlib import Path
import sys,re,hashlib,json
root=Path(__file__).resolve().parents[3];wt=root/'.cache/round4-answer';sys.path.insert(0,str(wt));sys.path.insert(0,str(wt/'tools'))
from check_mail_ood_rag import load_inputs,grade
from check_rag_facts import get
from mailsearch.generation import _requested_field, _is_related, _question_parts, _numbers, _names_supported
from argparse import ArgumentParser
p=ArgumentParser();p.add_argument('rev');p.add_argument('vk_port',type=int);p.add_argument('personal_port',type=int);a=p.parse_args()
coh=root/'.cache/ood-mail'/('combo-'+a.rev[:12]);
probes={key:re.compile(r,re.I) for key,r in {
 'place':r'где|мест[оае]|адрес\w*|кабинет\w*|в какой аудитории', 'time':r'во сколько|врем\w+|\bчас\w*|когда','date':r'како\w+ дат\w*|како\w+ числ\w*|како\w+ дн\w*|дата|дату', 'amount':r'цен[ауые]|стоимост\w*|сумм\w*|рубл\w*|плат[её]ж\w*|оплат\w*','person':r'\bкто\b|фамили\w+|имя\b|от кого|у кого|\bкем\b|\bкому\b','duration':r'длил\w*|занима\w*|долго\w*|продолж\w*|минут\w*|недел\w*','how':r'\bкак\b|каким образом|что сделать', 'which':r'\bкако\w*|\bкаки\w*|\bкака\w*|\bкаку\w*|\bкаке\w*','count':r'сколько|количеств\w+|числ[оа]','why':r'почему|причин\w*','code':r'код\w*|номер\w*','link':r'ссыл\w*|сайт\w*|страниц\w*|адрес\w*','file':r'файл\w*|влож\w*|прикреп\w*','subject':r'назван\w*|тема\w*|имен\w*|называ\w*','due':r'срок\w*|дедлайн\w*|крайн\w*','what':r'\bчто\b|чему\b|\bчем\b','reply':r'ответ\w*|\bпис[а-я]+|письм\w*','format':r'формат\w*|способ\w*|услови\w*','online':r'онлайн\w*|вебинар\w*|видео\w*|трансляци\w*','status':r'статус\w*|решен\w*|итог\w*','language':r'язык\w*|англий\w*|русск\w*'}.items()}
for spl in ['holdout','dev']:
 for source,port in [('personal',a.personal_port),('vk',a.vk_port)]:
  data=load_inputs(Path('/Users/kirill/Downloads/HackFPMI-mail-OOD'),source,spl)
  for idx,(cid,c) in enumerate((c,m) for c,m in data['cases'].items() if data['metadata'][c]['family']=='reviewed_semantic'):
   path=coh/spl/source/'after'/(hashlib.sha256(cid.encode()).hexdigest()+'.json');
   if not path.exists():continue
   orig=data['docs'][c['sources'][0]]; body=orig['doc']; question=c['query'];raw=json.loads(path.read_bytes());gold=data['metadata'][cid]; quals=grade(raw,c,gold,data['docs']);top=get(f'http://127.0.0.1:{port}','/api/search',{'mailbox':c['mailbox'],'date':c['date'],'query':question,'limit':5})['items'];place=next((i for i,r in enumerate(top,1) if r['id']==c['sources'][0]),None);patterns=[re.compile(p,re.I) for p in c['facts']]; print(spl,source,idx+1,quals['class'],'gold_position',place,'related',_is_related(question,body),'field',_requested_field(question),'facets',[k for k,p in probes.items() if p.search(question)],'query_tokens',len(question.split()),'evidence_tokens',len(c['evidence'].split()),'answer_verified_cites',str(quals['supported_citations'])+'/'+str(quals['cited']),'value_shape',[(sum(ch.isdigit() for ch in p.pattern),sum(ch.isalpha() for ch in p.pattern), bool(p.search(raw['answer']))) for p in patterns]);
