from pathlib import Path
from datetime import date
import sys,json,collections,time,statistics,re
import numpy as np,onnxruntime as ort
from tokenizers import Tokenizer
root=Path.cwd()
sys.path.insert(0,str(root))
from mailsearch.dates import parse_date_scope
from mailsearch.store import load_jsonl
p=Path.home()/'Downloads/HackFPMI-mail-OOD/suite'
model=Path.home()/'.cache/huggingface/hub/models--cross-encoder--mmarco-mMiniLMv2-L12-H384-v1/snapshots/1427fd652930e4ba29e8149678df786c240d8825'
opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1
session=ort.InferenceSession(str(model/'onnx/model_qint8_arm64.onnx'),opt,providers=['CPUExecutionProvider'])
tok=Tokenizer.from_file(str(model/'tokenizer.json'));tok.enable_truncation(max_length=208);tok.enable_padding(pad_id=1,pad_token='<pad>')
keys={i.name for i in session.get_inputs()}; outputs=[];lat=[]
ws=re.compile(r'https?://\S+|[A-Za-z0-9_.!+-]+@[\w.-]+|(?:syn-|l-)[a-f0-9]{8,}',re.I)
clean=lambda t:' '.join(ws.sub(' ',t).split())
for name in ['vk_dev','personal_dev','vk_holdout','personal_holdout']:
 ds=p/name;docs=load_jsonl(ds/'documents.jsonl');qs={q['query_id']:q for f in ['dev','empty'] for q in load_jsonl(ds/f'queries_{f}.jsonl')};rels=collections.defaultdict(set)
 for r in load_jsonl(ds/'qrels_dev.jsonl'):
  if r['relevance']>0:rels[r['query_id']].add(r['doc_id'])
 cases=[r for r in load_jsonl(ds/'cases.jsonl') if r['judgment']=='assistant_reviewed' and not r.get('variant')]
 ranks=json.load(open(root/'.cache/round4/round2'/name/'after/rankings.json'))['queries']
 for c in cases:
  q=qs[c['query_id']];s=parse_date_scope(q['query'],date.fromisoformat(q['query_date']));qtext=s.text.strip(' .?') or q['query'];candidates=[d for d in docs if d['synthetic_user']==q['synthetic_user'] and (not d['doc_dt'] or d['doc_dt']<=q['query_date']) and (not s.bounded or d['doc_dt'] and (not s.start or d['doc_dt']>=s.start.isoformat())and (not s.end or d['doc_dt']<=s.end.isoformat()))]; gold=rels[q['query_id']];found=ranks[q['query_id']]['ranked_ids']
  if not 0<len(candidates)<=24:continue
  b=[];di=[]
  for d in candidates:
   t=clean(d['doc']);title,tick,body=t.partition('ТЕКСТ ПИСЬМА:');body=body if tick else t
   for z in [0]+([600] if len(body)>700 else []):b.append((qtext,((title[:150]+' ')if tick else '')+body[z:z+870]));di.append(d['doc_id'])
  at=time.perf_counter(); scores=collections.defaultdict(lambda: float('-inf'))
  for i in range(0,len(b),8):
   enc=tok.encode_batch(b[i:i+8]); inputs={'input_ids':np.array([e.ids for e in enc]),'attention_mask':np.array([e.attention_mask for e in enc]),'token_type_ids':np.array([e.type_ids for e in enc])}
   raw=session.run(None,{k:v for k,v in inputs.items() if k in keys})[0].ravel()
   for did,l in zip(di[i:i+8],raw): scores[did]=max(scores[did],float(l))
  lat.append(1000*(time.perf_counter()-at)); goldscore=max((v for i,v in scores.items() if i in gold),default=None); top=max(scores,key=scores.get)
  outputs.append(dict(source=name,old_success=bool(set(found)&gold),old_return=bool(found),has_gold=bool(gold),top_gold=top in gold,max=scores[top],gold_score=goldscore,candidates=len(scores)))
print('reviewed',len(outputs),'ms',np.quantile(lat,[.5,.95,.99]).round(1).tolist())
for group in ('positive_returned','missed_positive','verified_empty'):
 a=[x for x in outputs if (group=='verified_empty' and not x['has_gold'] or group=='missed_positive' and x['has_gold'] and not x['old_success'] or group=='positive_returned' and x['has_gold'] and x['old_success'])]
 print(group,len(a),'top correct',sum(x['top_gold'] for x in a),'max',np.quantile([x['max']for x in a],[0,.25,.5,.75,1]).round(2).tolist()if a else [])
 for t in (-3,0,2,4,6):print(t, sum(x['max']>t for x in a),sum(x['top_gold']and x['max']>t for x in a))
(root/'.cache/round4/cross_trial.json').write_text(json.dumps(outputs))
