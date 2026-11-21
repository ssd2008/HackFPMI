from pathlib import Path
from urllib.request import urlopen, Request
import subprocess
import time
import json
import sys
root = Path.cwd()
work = root / ".cache/iteration"
urls = {"before": "http://127.0.0.1:8765", "after": "http://127.0.0.1:18769"}
for url in urls.values():
    for retry in range(75):
        try:
            with urlopen(url + "/api/mailboxes", timeout=2) as resp:
                assert len(json.load(resp)["mailboxes"]) == 12
            break
        except OSError:
            time.sleep(1)
    else:
        raise SystemExit("Server unavailable: " + url)
    for raw in (root / "data/queries_dev.jsonl").read_text().splitlines()[:5]:
        q = json.loads(raw)
        body = {"mailbox": q["synthetic_user"], "query": q["query"], "date": q["query_date"], "limit": 10}
        req = Request(url + "/api/search", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urlopen(req) as resp:
            json.load(resp)
single = ["--k", "1", "5", "10", "--repeats", "5"]
parallel = ["--k", "10", "--concurrency", "8", "--repeats", "5"]
wide = ["--k", "1", "5", "10", "20", "50", "--repeats", "1"]
jobs = [
    ("one", ["before", "after"], single),
    ("eight", ["after", "before"], parallel),
    ("open", ["before", "after"], parallel + ["--queries", "tests/fixtures/queries_no_date.jsonl", "--unjudged"]),
    ("wide", ["after", "before"], wide),
    ("para", ["before", "after"], wide + ["--queries", "tests/fixtures/queries_paraphrased.jsonl"]),
    ("typo", ["after", "before"], wide + ["--queries", "tests/fixtures/queries_typos.jsonl"]),
]
for name, versions, flags in jobs:
    for version in versions:
        dst = work / f"r2-{version}-{name}.json"
        with dst.with_suffix(".log").open("w") as log:
            subprocess.run([sys.executable, "benchmark.py", "--url", urls[version], "--output", str(dst), *flags], stdout=log, stderr=subprocess.STDOUT, check=True)
        data = json.loads(dst.read_text())
        print(version, name, "p99", data["timing"]["server_ms"]["p99"], "nDCG@10", data["quality_first_pass"].get("10", {}).get("ndcg"), flush=True)
