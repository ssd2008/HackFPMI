"""Attach the frozen Git corpora to the available evaluation splits without duplicating messages."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo
from import_mail_export import write_corpus

def sha(path):
    with path.open("rb") as source:return hashlib.file_digest(source,"sha256").hexdigest()
def place(source,target,wanted):
    if sha(source)!=wanted:raise ValueError(f"Different corpus at {source}")
    if target.exists():
        if sha(target)!=wanted:raise ValueError(f"A different file already exists at {target}")
        return
    try:os.link(source,target)
    except OSError:
        with target.open("xb") as out,source.open("rb") as inp:shutil.copyfileobj(inp,out)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True,help="Git directory containing documents.jsonl and both mail exports.")
    parser.add_argument("--suite-root",type=Path,default=Path(__file__).resolve().parents[1]/"suite")
    a=parser.parse_args();dirs=sorted(p for p in a.suite_root.glob("*") if (p/"manifest.json").is_file())
    if not dirs:raise SystemExit("No suite manifests found")
    sources=a.data.resolve()
    with TemporaryDirectory(dir=a.suite_root) as temp:
        stage=Path(temp);personal=stage/"personal.jsonl"
        for p in dirs:
            digest=json.loads((p/"manifest.json").read_text())["sha256"]["documents.jsonl"]
            target=p/"documents.jsonl"
            if target.exists():
                if sha(target)!=digest:raise ValueError(f"Different local corpus at {target}")
                continue
            if p.name.startswith("vk_"):source=sources/"documents.jsonl"
            elif p.name.startswith("personal_"):
                if not personal.exists():
                    with (sources/"Sergey-mails.jsonl").open("rb") as raw:
                        write_corpus(raw,stage/"sergey","independent-export-01",ZoneInfo("Europe/Moscow"))
                    with personal.open("xb") as out:
                        for name in (sources/"Kirill-mails.jsonl",stage/"sergey/documents.jsonl"):
                            with name.open("rb") as inp:shutil.copyfileobj(inp,out)
                source=personal
            else:continue
            place(source,target,digest)
    print("Snapshots verified for "+", ".join(p.name for p in dirs))
if __name__=="__main__":main()
