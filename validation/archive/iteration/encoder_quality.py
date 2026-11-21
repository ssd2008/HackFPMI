from datetime import date
from pathlib import Path
import importlib.util
import json
import statistics
import sys

root=Path(__file__).resolve().parents[2]
sys.path.extend([str(root/".cache/iteration/ortlib"),str(root)])
import onnxruntime as ort
import numpy as np
from huggingface_hub import hf_hub_download
from mailsearch import E5_MODEL,E5_REVISION
from mailsearch.store import load_jsonl,DATA,CACHE
from mailsearch.retrieval import SearchEngine
from benchmark import metrics,load_benchmark

engine=SearchEngine(load_jsonl(DATA/'documents.jsonl'),CACHE)
tokenizer=engine.dense.tokenizer
eager=engine.dense.query
sets=[load_benchmark(DATA),load_benchmark(DATA,root/'tests/fixtures/queries_paraphrased.jsonl')]
def test(label):
    for i,(_,queries,rels,_) in enumerate(sets):
        scores={k:[] for k in (1,5,10)}
        for row in queries:
            ranks=[pair[0] for pair in engine.search(row['synthetic_user'],row['query'],date.fromisoformat(row['query_date']),10)]
            for k in scores: scores[k].append(metrics(ranks,rels[row['query_id']], k))
        print(label,i,{k:{metric:round(statistics.mean(v[metric] for v in values if metric in v),6) for metric in ('ndcg','recall','mrr_single')} for k,values in scores.items()},flush=True)
test('pytorch')
for file in ['onnx/model_qint8_avx512_vnni.onnx','onnx/model.onnx']:
    opt=ort.SessionOptions()
    opt.intra_op_num_threads=2
    opt.add_session_config_entry("session.intra_op.allow_spinning","0")
    session=ort.InferenceSession(hf_hub_download(E5_MODEL,file,revision=E5_REVISION,local_files_only=True),opt,providers=['CPUExecutionProvider'])
    def predict(s):
        x=tokenizer(['query: '+s],max_length=120,truncation=True,return_tensors='np')
        last=session.run(None,{'input_ids':x.input_ids,'attention_mask':x.attention_mask,'token_type_ids':np.zeros_like(x.input_ids)})[0][0]
        summed=last.sum(0)
        return summed/np.linalg.norm(summed)
    engine.dense.query=predict
    test(file)
