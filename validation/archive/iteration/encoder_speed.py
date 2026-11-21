import json
import random
import statistics
import sys
from pathlib import Path
from time import perf_counter

root = Path(__file__).resolve().parents[2]
sys.path.extend([str(root / ".cache/iteration/ortlib"), str(root)])
import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel
from mailsearch import E5_MODEL, E5_REVISION

tok = AutoTokenizer.from_pretrained(E5_MODEL, revision=E5_REVISION, local_files_only=True)
values = ["платёж не прошёл", "отказали в оплате картой", "уведомление о завершённой поставке",
          "приехал водитель чтобы забрать контейнер с площадки", "коммунальные услуги квитанции",
          "индивидуальный тариф на возврат пятой партии", "термокружка протекла в машине что с обменом",
          "кто отправил последние документы на согласование", "откуда должен прийти перевод денег",
          "искали поврежденный заказ где покупатель просил продавца вернуть оплату за этот товар"]
def run(label, encode, reference=None):
    times = []
    probes = [encode("query: " + q) for q in values]
    for i in range(80):
        start = perf_counter()
        encode("query: " + values[i % len(values)] + " " + str(i % 23))
        times.append((perf_counter() - start) * 1000)
    print(label, "p50", round(statistics.median(times),3),"p95", round(np.percentile(times,95),3),
          "min cosine", round(min((np.dot(a,b) for a,b in zip(probes,reference)),default=1),6)
          if reference is not None else "ref",flush=True)
    return probes
model=AutoModel.from_pretrained(E5_MODEL,revision=E5_REVISION,local_files_only=True).eval()
def norm(last, mask):
    sums=(last*mask[...,None]).sum(1)[0]
    return sums / np.linalg.norm(sums)
@torch.inference_mode()
def pyt(query):
    a=tok([query],return_tensors='pt')
    return norm(model(**a).last_hidden_state.detach().numpy(),a.attention_mask.numpy())
torch.set_num_threads(4)
base=run("pytorch/4",pyt)
for threads in [2,6]:
    torch.set_num_threads(threads)
    run("pytorch/"+str(threads),pyt,base)
for file in ['onnx/model_qint8_avx512_vnni.onnx','onnx/model.onnx']:
    path=hf_hub_download(E5_MODEL,file,revision=E5_REVISION,local_files_only=True)
    for threads in [2,4,6]:
        opt=ort.SessionOptions()
        opt.intra_op_num_threads=threads
        opt.inter_op_num_threads=1
        opt.add_session_config_entry("session.intra_op.allow_spinning","0")
        session=ort.InferenceSession(path,opt,providers=['CPUExecutionProvider'])
        names=[n.name for n in session.get_inputs()]
        print(file, names, [n.name for n in session.get_outputs()],flush=True)
        def predict(s):
            inputs=tok([s],return_tensors='np')
            feeds={n:inputs[n] if n in inputs else np.zeros_like(inputs.input_ids) for n in names}
            last=session.run(None,feeds)[0]
            return norm(last, inputs.attention_mask)
        run(file+'/'+str(threads),predict,base)
        del session
