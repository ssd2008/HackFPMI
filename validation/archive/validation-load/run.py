from pathlib import Path
import json, sys, subprocess, time
from urllib.request import urlopen
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from benchmark import fetch
work = ROOT / ".cache/validation-load"
urls = {"baseline":"http://127.0.0.1:18773", "candidate":"http://127.0.0.1:8766"}
queries = [json.loads(line) for line in (ROOT / "data/queries_dev.jsonl").open() if line.strip()]
scenarios = {"public_1_client":[],"public_8_clients":["--concurrency","8"],
             "no_receipt_day_8_clients":["--concurrency","8","--queries","tests/fixtures/queries_no_date.jsonl","--unjudged"]}
output={"corpus":"VK public, 16376 letters","ranking_baseline":"5fe3f5d3739ac523d0b4088f15eef4b50391dd28",
        "ranking_candidate_sha256":"d747e72ddb05772213a7042f8cc5372e07b2d274b9c45ed0cda4db7bba56ac29",
        "method":"Two subsequent series at 1089 HTTP requests per version per scenario, matching shuffled orders, alternating version order. One text per original query per pass. No query/result caches. First request after each service startup includes online search but excludes server initialization; queries may be encoded by the application as a startup warm-up.",
        "machine":"Apple M5 Pro, 48 GiB, Python 3.12.13","k":10,"nearest_rank":True,"repeats_of_99_queries":11,
        "first_external_search":{},"scenarios":{s:[] for s in scenarios}}
for version,url in urls.items():
    for retry in range(60):
        try:
            with urlopen(url+"/api/mailboxes", timeout=3) as r:
                assert sum(x["count"] for x in json.load(r)["mailboxes"])==16376
            break
        except OSError:time.sleep(1)
    else:raise RuntimeError("Server unavailable: "+version)
    _,server,wall=fetch(url,queries[0],10)
    output["first_external_search"][version]={"server_ms":round(server,3),"client_ms":round(wall,3),"query_id":queries[0]["query_id"]}
    for q in queries[1:6]:fetch(url,q,10)
for series in range(2):
    for i,(scenario,flags) in enumerate(scenarios.items()):
        row={}
        for version in (("baseline","candidate") if (series+i)%2==0 else ("candidate","baseline")):
            path=work/f"{series+1}-{scenario}-{version}.json"
            with path.with_suffix(".log").open("w") as out:
                subprocess.run([sys.executable,"benchmark.py","--url",urls[version],"--k","10","--repeats","11","--output",str(path),*flags],stdout=out,stderr=subprocess.STDOUT,check=True)
            t=json.loads(path.read_text())["timing"]
            assert t["search_calls"]==1089
            row[version]={"calls":t["search_calls"],"http_errors":0,"timeouts":0,"server_ms":t["server_ms"],"loopback_http_ms":t["loopback_http_ms"],"rank_changes":t["ranking_changes_on_repeats"]}
        output["scenarios"][scenario].append(row)
        print(series+1,scenario,[r["server_ms"]["p99"] for r in row.values()],flush=True)
(ROOT/"results/validation_load.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
