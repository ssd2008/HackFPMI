"""Make interoperable evaluation subsets from public human qrels without inventing mailboxes or receipt dates."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"github/datasets/local";OUT=ROOT/"external/benchmarks";ASOF="2026-11-21"
def encode(row):return json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n"
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,rows):
    with path.open("w",encoding="utf-8") as stream:
        for row in rows:stream.write(encode(row))
def parquet(p):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(p).iter_batches(batch_size=1500):
        yield from batch.to_pylist()
def tsv(path):
    opener=gzip.open if str(path).endswith(".gz") else open
    with opener(path,"rt",encoding="utf-8") as f:
        for ln in f:
            if ln.strip():yield ln.rstrip("\r\n").split("\t")
def trec(links):
    labels=defaultdict(dict)
    for fields in links:
        if len(fields)==1:fields=fields[0].split()
        if len(fields)!=4:raise ValueError(f"Expected four qrel fields, got {len(fields)}")
        q,_,d,g=fields;labels[q][d]=max(int(g),labels[q].get(d,0))
    return labels
def pairs(path):return {r[0]:"\t".join(r[1:]) for r in tsv(path)}
def json_corpus(paths):
    for path in paths:
        with gzip.open(path,"rb") as f:
            for line in f:
                row=json.loads(line)
                i=next((row[k] for k in ("docid","doc_id","id","_id") if k in row),None)
                if i is None:raise ValueError(str(row.keys()))
                text=row.get('text') or row.get('contents') or row.get('passage')
                if not isinstance(text,str):raise ValueError(str(row.keys()))
                yield str(i), (row.get("title") or "")+"\n"+text
def tsv_collection(lines,large_docs=False):
    for line in lines:
        i,sep,body=line.rstrip(b"\r\n").partition(b"\t")
        if not sep:continue
        if large_docs:
            url,sep2,title_body=body.partition(b"\t")
            if sep2:body=title_body.replace(b"\t",b"\n",1)
        yield i.decode("utf-8"),body
def gz_tsv(path):
    with gzip.open(path,"rb") as f:yield from tsv_collection(f,True)
def local_tsv(path):
    with path.open("rb") as f:yield from tsv_collection(f)
def tar_items(path,accept):
    with tarfile.open(path,"r|gz",bufsize=1024*1024) as tar:
        for obj in tar:
            name=obj.name.lstrip("./")
            if obj.isfile() and accept(name):
                with tar.extractfile(obj) as f:yield name,f
def map_archive(path,needed):
    out={}
    for name,f in tar_items(path,lambda n:n.rsplit("/",1)[-1] in needed):
        key=name.rsplit("/",1)[-1];out[key]=[l.decode("utf-8").rstrip("\r\n").split("\t") for l in f if l.strip()]
        if len(out)==len(needed):break
    if not needed.issubset(out):raise ValueError(f"Missing archive entries: {needed-set(out)}")
    return out
def tar_collection(path):
    for _,f in tar_items(path,lambda n:n.endswith("collection.tsv")):
        yield from tsv_collection(f);return
    raise ValueError("Collection missing")
def read_lotte_native(src,kind,q,l):
    for ln in src:
        if not ln.strip():continue
        j=json.loads(ln);qid=kind+"-"+str(j["qid"]);q[qid]=j["query"]
        for doc in j["answer_pids"]:l[qid][str(doc)]=1
def setup(name):
    src=RAW/name
    if name=="rubq-2.0":
        qs=next((src/"questions").rglob("test-*.parquet"));ps=next((src/"paragraphs").rglob("paragraphs-*.parquet"))
        q={};l=defaultdict(dict)
        for v in parquet(qs):
            i=str(v["uid"]);q[i]=v["question_text"]
            for d in (v["paragraphs_uids"] or {}).get("with_answer") or []:l[i][str(d)]=1
        corpus=((str(v["uid"]),v["text"]) for v in parquet(ps))
        return q,l,corpus,"RuBQ 2.0 test, paragraphs_uids.with_answer",100000,False
    if name=="beir-scifact":
        qs=next((src/"hf/queries").glob("*.parquet"));ds=next((src/"hf/corpus").glob("*.parquet"));q={str(v["_id"]):v["text"] for v in parquet(qs)}
        l=defaultdict(dict);it=iter(tsv(src/"hf-qrels/test.tsv"));header=next(it)
        if header[:2]!=["query-id","corpus-id"]:raise ValueError(str(header))
        for i,d,g in it:l[i][d]=int(g)
        return q,l,((str(v["_id"]),v.get("title","")+"\n"+v["text"]) for v in parquet(ds)),"BEIR SciFact test",100000,False
    if name=="miracl-ru":
        topics=src/"topics/miracl-v1.0-ru"
        q=pairs(topics/"topics/topics.miracl-v1.0-ru-dev.tsv");l=trec(tsv(topics/"qrels/qrels.miracl-v1.0-ru-dev.tsv"))
        parts=sorted((src/"corpus/miracl-corpus-v1.0-ru").glob("docs-*.jsonl.gz"))
        if len(parts)!=20:raise ValueError(f"Need 20 MIRACL shards, have {len(parts)}")
        return q,l,json_corpus(parts),"MIRACL Russian dev, public qrels",20000,True
    if name=="mr-tydi-ru":
        topics=src/"topics/mrtydi-v1.1-russian/ir-format-data";q=pairs(topics/"topics.test.txt");l=trec(tsv(topics/"qrels.test.txt"))
        return q,l,json_corpus([src/"corpus/mrtydi-v1.1-russian/corpus.jsonl.gz"]),"Mr. TyDi v1.1 Russian test",20000,True
    if name=="mmarco-ru":
        q=pairs(src/"data/google/queries/dev/russian_queries.dev.small.tsv");l=trec(tsv(src/"data/qrels.dev.small.tsv"))
        return q,l,local_tsv(src/"data/google/collections/russian_collection.tsv"),"mMARCO Russian Google Translate dev.small",22000,False
    if name=="msmarco-passage":
        a=src/"collectionandqueries.tar.gz";needed={"queries.dev.small.tsv","qrels.dev.small.tsv"};items=map_archive(a,needed)
        q={row[0]:"\t".join(row[1:]) for row in items["queries.dev.small.tsv"]};l=trec(items["qrels.dev.small.tsv"])
        return q,l,tar_collection(a),"MS MARCO Passage v1 dev.small",22000,False
    if name=="msmarco-doc":
        q=pairs(src/"msmarco-docdev-queries.tsv.gz");l=trec(tsv(src/"msmarco-docdev-qrels.tsv.gz"))
        return q,l,gz_tsv(src/"msmarco-docs.tsv.gz"),"MS MARCO Document v1 dev",10000,False
    if name=="lotte":
        q={};l=defaultdict(dict)
        # The pooled set contains all five domains; including them individually again would multiply identical questions.
        folder=src/"hf/pooled/test"
        if (folder/"collection.tsv").is_file() and all((folder/("qas."+x+".jsonl")).exists() for x in ("forum","search")):
            for kind in ("forum","search"):
                with (folder/("qas."+kind+".jsonl")).open() as f:read_lotte_native(f,kind,q,l)
            corpus=local_tsv(folder/"collection.tsv")
        else:
            a=src/"lotte.tar.gz"
            for n,f in tar_items(a,lambda n:"pooled/test/qas." in n and n.endswith(".jsonl")):read_lotte_native(f,n.rsplit(".",2)[1],q,l)
            def from_tar():
                for _,f in tar_items(a,lambda n:n.endswith("pooled/test/collection.tsv")):
                    yield from tsv_collection(f);return
                raise ValueError("Missing pooled test corpus")
            corpus=from_tar()
        if not q:raise ValueError("Missing LoTTE test questions")
        return q,l,corpus,"LoTTE pooled test, forum and search",22000,False
    raise ValueError(name)
def build(name):
    queries,labels,corpus,source,limit,neighbors=setup(name)
    usable={q for q in queries if any(g>0 for g in labels.get(q,{}).values())}
    if not usable:raise ValueError("No positive judged queries")
    req={d for q in usable for d in labels[q]}
    wanted=set(req)
    if neighbors:
        for d in req:
            page,sep,s=d.rpartition("#")
            if sep and s.isdigit():wanted.update(page+"#"+str(int(s)+j) for j in (-2,-1,1,2) if int(s)+j>=0)
    out=OUT/name;out.mkdir(parents=True,exist_ok=True)
    prefix=name+"/";box="collection-"+name
    def row(id_,text):
        if isinstance(text,bytes):text=text.decode("utf-8")
        return {"doc_id":prefix+id_,"synthetic_user":box,"doc":text.strip(),"doc_dt":None}
    selected=set();reservoir=[];randomizer=random.Random(240114);n=other=0
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=out,prefix=".corpus-",delete=False) as f:
        temp=Path(f.name)
        for id_,text in corpus:
            n+=1
            if id_ in wanted:
                if id_ in selected:raise ValueError("Duplicate annotated document "+id_)
                f.write(encode(row(id_,text)));selected.add(id_);continue
            other+=1
            if len(reservoir)<limit:reservoir.append((id_,text))
            else:
                j=randomizer.randrange(other)
                if j<limit:reservoir[j]=(id_,text)
        missing=req-selected
        if missing:
            temp.unlink(missing_ok=True);raise ValueError(f"{name}: {len(missing)} missing judgment references: {sorted(missing)[:6]}")
        for i,text in reservoir:
            if i in selected:raise ValueError("Duplicate corpus ID "+i)
            selected.add(i);f.write(encode(row(i,text)))
    temp.replace(out/"documents.jsonl")
    ids=sorted(usable)
    write(out/"queries_dev.jsonl",({"query_id":prefix+"q/"+i,"synthetic_user":box,"query":queries[i],"query_date":ASOF} for i in ids))
    write(out/"qrels_dev.jsonl",({"query_id":prefix+"q/"+q,"doc_id":prefix+d,"relevance":g} for q in ids for d,g in sorted(labels[q].items())))
    info={"source":source,"language":"ru" if name in ("rubq-2.0","miracl-ru","mr-tydi-ru","mmarco-ru") else "en",
          "raw_corpus_rows":n,"documents":len(selected),"complete_corpus":n==len(selected),"queries":len(ids),"qrels":sum(len(labels[q]) for q in ids),
          "unique_judged_documents":len(req),"adjacent_unjudged_documents":len(selected)-len(req)-len(reservoir),"random_unjudged_documents":len(reservoir),
          "query_pool":"Original published judgments with at least one positive; restricted to the indicated split.",
          "raw_to_local_id":"Prefix raw document IDs with "+prefix+"; prefix raw question IDs with "+prefix+"q/. LoTTE namespaced forum and search.",
          "sha256":{p.name:sha(p) for p in sorted(out.glob("*.jsonl"))}}
    (out/"manifest.json").write_text(json.dumps(info,ensure_ascii=False,indent=2)+"\n")
    print(name,n,len(selected),len(ids),info["qrels"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("dataset",nargs="+",choices=("rubq-2.0","beir-scifact","miracl-ru","mr-tydi-ru","mmarco-ru","msmarco-passage","msmarco-doc","lotte"));a=p.parse_args()
    for name in a.dataset:build(name)
