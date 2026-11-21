"""Choose a fixed 4:4:2 panel for each split using only task metadata."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def key(text):return hashlib.sha256(("mail-panel-v1/"+text).encode()).digest()
def panel(rows):
    out=[]
    for kind,wanted in (("known_item",100),("collection",100),("empty",50)):
        eligible=[r for r in rows if r["task_type"]==kind]
        # Keep all independently inspected day windows, then rotate other families and mailboxes.
        chosen=sorted((r for r in eligible if r["judgment"]=="assistant_reviewed"),key=lambda r:key(r["query_id"]))[:wanted]
        used={r["query_id"] for r in chosen}
        buckets=defaultdict(list)
        for r in eligible:
            if r["query_id"] not in used:buckets[r["family"],r["family_id"].split(":",1)[0]].append(r)
        for bucket in buckets.values():bucket.sort(key=lambda r:key(r["query_id"]))
        while len(chosen)<wanted:
            nonempty=[k for k in buckets if buckets[k]]
            if not nonempty:break
            for b in sorted(nonempty,key=lambda k:key(repr(k))):
                chosen.append(buckets[b].pop())
                if len(chosen)==wanted:break
        out.extend(chosen)
    return sorted(out,key=lambda r:r["query_id"])
def main():
    target=ROOT/"panels";target.mkdir(exist_ok=True)
    for folder in sorted((ROOT/"suite").glob("*")):
        if not (folder/"cases.jsonl").exists():continue
        with (folder/"cases.jsonl").open() as f:rows=[r for r in map(json.loads,f) if not r.get("variant")]
        cases=panel(rows); path=target/(folder.name+".jsonl")
        with path.open("w") as out:
            for r in cases:
                out.write(json.dumps({k:r[k] for k in ("query_id","family","task_type","judgment")},ensure_ascii=False)+"\n")
        print(folder.name,len(cases))
if __name__=="__main__":main()
