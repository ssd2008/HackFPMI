from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from mailsearch import generation as g
read=lambda path:[json.loads(i) for i in path.read_text().splitlines()]
fix=ROOT/'tests/fixtures/validation_pilot'
ids=['vk-pilot-01','vk-pilot-02','vk-pilot-03']
qs={q['query_id']:q for q in read(fix/'queries.jsonl') if q['query_id'] in ids}
before=json.loads((ROOT/'results/validation_pilot/after.json').read_text())['queries']
docs={d['doc_id']:d for d in read(ROOT/'data/documents.jsonl')}
out={}
for qid in ids:
 q=qs[qid]['query'];ds=[docs[k] for k in before[qid]['ranked_ids'][:5]]; st=time.perf_counter()
 result=g.answer_question(q,ds)
 print(qid,'rel',[g._is_related(q,d['doc']) for d in ds],g._question_parts(q),result,flush=True)
 out[qid]={'input_ranked_ids':[d['doc_id'] for d in ds],'result':result,'wall_ms':round((time.perf_counter()-st)*1000)}
(ROOT/'.cache/validation-pilot/rag-dev-fixed.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
