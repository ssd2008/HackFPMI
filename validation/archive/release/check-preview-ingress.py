import json
from pathlib import Path
import subprocess
import urllib.parse

cache=Path(__file__).parent
origin="https://activation-suggests-yields-frontier.trycloudflare.com"
page="https://ssd2008.github.io"
GET_HEADERS=["-H","Origin: "+page]
def req(path,*options,tag):
    meta=cache/(tag+".headers"); payload=cache/(tag+".json")
    cmd=["/usr/bin/curl","--silent","--show-error","--connect-timeout","8","--max-time","30","--dump-header",str(meta),"--output",str(payload),"--write-out",'%{http_code} %{ssl_verify_result} %{remote_ip}',*GET_HEADERS,*options,origin+path]
    vals=subprocess.check_output(cmd,text=True).split()
    headers={line.split(':',1)[0].strip().lower():line.split(':',1)[1].strip() for line in meta.read_text().splitlines() if ':' in line}
    return int(vals[0]),vals[1],vals[2],headers,payload
status,tls,peer,head,mb=req('/api/mailboxes',tag='public-boxes')
print('public boxes status',status,'TLS',tls,'peer',peer,'edge',head.get('cf-ray'),'cors',head.get('access-control-allow-origin'),flush=True)
assert status==200 and tls=='0' and head.get('access-control-allow-origin')==page
boxes=json.loads(mb.read_bytes());items=boxes if isinstance(boxes,list) else boxes.get('items',boxes.get('mailboxes',[]));print('boxes',len(items),'keys',list(boxes) if isinstance(boxes,dict) else '-');assert len(items)==12
q=json.loads((Path('data')/'queries_dev.jsonl').open().readline());request=json.dumps(dict(mailbox=q['synthetic_user'],query=q['query'],date=q['query_date'],limit=1),ensure_ascii=False)
status,_,_,_,hit=req('/api/search','-H','Content-Type: application/json','--data-binary',request,tag='public-search');assert status==200
hit=json.loads(hit.read_bytes())['items'][0];identifier=hit.get('doc_id') or hit.get('id');assert identifier
url='/api/mail/'+urllib.parse.quote(identifier,safe='')+'?'+urllib.parse.urlencode({'mailbox':q['synthetic_user'],'date':q['query_date']})
status,_,_,_,message=req(url,tag='public-message');assert status==200;fields=json.loads(message.read_bytes());assert (fields.get('doc_id') or fields.get('id'))==identifier
status,_,_,cors,_=req('/api/search','--request','OPTIONS','-H','Access-Control-Request-Method: POST','-H','Access-Control-Request-Headers: authorization,x-mailsearch-visitor,content-type',tag='public-preflight')
out=dict(origin=origin,api_tls_valid=True,https_peer=peer,cloudflare_ray=head.get('cf-ray'),pages_origin=page,mailboxes=len(items),public_sha=boxes.get('corpus_sha256') if isinstance(boxes,dict) else None,synthetic_search_works=True,open_message_works=True,session_preflight_http=status,session_preflight_allow_headers=cors.get('access-control-allow-headers'),session_preflight_allow_origin=cors.get('access-control-allow-origin'))
(cache/'preview-external-smoke.json').write_text(json.dumps(out,indent=2)+'\n');print('public search and message verified;',out)
