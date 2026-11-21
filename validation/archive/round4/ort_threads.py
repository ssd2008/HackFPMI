import concurrent.futures,json,statistics,time,sys,random
from pathlib import Path
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
sys.path.insert(0,str(Path.cwd()))
from mailsearch import E5_MODEL,E5_REVISION,E5_QUERY_GRAPH
tok=AutoTokenizer.from_pretrained(E5_MODEL,revision=E5_REVISION,local_files_only=True)
graph=hf_hub_download(E5_MODEL,E5_QUERY_GRAPH,revision=E5_REVISION,local_files_only=True)
samples=[x['query'] for x in map(json.loads,open('data/queries_dev.jsonl'))]
samples += [x['query'] for x in map(json.loads,(Path.home()/'Downloads/HackFPMI-mail-OOD/suite/vk_dev/queries_dev.jsonl').open())][:100]
random.Random(27021).shuffle(samples)
feeds=[dict(tok('query: '+s,return_tensors='np',return_token_type_ids=True,truncation=True,max_length=120)) for s in samples]
sessions={}
for nt in [1,2]:
 s=ort.SessionOptions();s.intra_op_num_threads=nt;s.inter_op_num_threads=1;s.add_session_config_entry('session.intra_op.allow_spinning','0')
 sessions[nt]=ort.InferenceSession(graph,s,providers=['CPUExecutionProvider'])
 for f in feeds[:8]:sessions[nt].run(None,f)
for clients in [1,8]:
 for nt in [2,1,1,2]:
  def run(f):
   t=time.perf_counter_ns();sessions[nt].run(None,f);return (time.perf_counter_ns()-t)/1e6
  with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as pool:
   ms=list(pool.map(run,feeds))
  print(clients,nt,'median / p95 / p99',np.round(np.quantile(ms,[.5,.95,.99]),2),flush=True)
