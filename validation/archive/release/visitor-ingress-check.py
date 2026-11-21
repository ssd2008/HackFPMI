import json
import urllib.request
from urllib.error import HTTPError
from urllib.parse import urlencode
from sys import argv

base = argv[1].rstrip('/')
origin = 'https://ssd2008.github.io'
allow = {'Origin': origin, 'X-Mailsearch-Visitor': '1'}
def request(method, path, body=None, headers=allow):
    out = None if body is None else json.dumps(body).encode()
    hdr = ({'Content-Type': 'application/json'} if out is not None else {}) | headers
    req = urllib.request.Request(base+path, data=out, headers=hdr, method=method)
    try: response = urllib.request.urlopen(req, timeout=20)
    except HTTPError as error: response = error
    raw = response.read()
    return response.status, response.headers, (json.loads(raw) if response.headers.get_content_type() == 'application/json' else None)

path = '/api/visitor/session'
preflight = request('OPTIONS',path,headers={'Origin':origin,'Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'authorization,content-type,x-mailsearch-visitor'})
assert preflight[0] == 200 and preflight[1]['access-control-allow-origin'] == origin and 'no-store' in preflight[1]['cache-control']
allowed_headers = {h.strip().lower() for h in preflight[1]['access-control-allow-headers'].split(',')}
assert {'authorization','x-mailsearch-visitor','content-type'} <= allowed_headers
status,headers,listing = request('GET','/api/mailboxes')
assert status == 200 and headers['access-control-allow-origin'] == origin and listing['visitor_mail']
assert len(listing['mailboxes']) == 12 and sum(b['count'] for b in listing['mailboxes'])==16376 and listing['corpus_sha256']=='06feb4791a52faeb777f1a8d4c22d99efd26a279943a19bdfbd0b14bc9c8daa5'
assert request('POST',path,{}, dict(allow,Origin='https://elsewhere.example'))[0] == 403
unauthorized=request('GET','/api/visitor/accounts')
assert unauthorized[0] == 401 and unauthorized[1]['access-control-allow-origin'] == origin
sessions=[]
try:
    for i in range(2):
        status,headers,created=request('POST',path,{})
        assert status==200 and created['expires_in']==3600 and 'no-store' in headers['cache-control'] and headers['access-control-allow-origin'] == origin
        sessions.append(dict(allow, Authorization='Bearer '+created['token']))
        status,h,accounts=request('GET','/api/visitor/accounts',headers=sessions[-1])
        assert status==200 and h['access-control-allow-origin'] == origin and accounts['accounts']==[] and accounts['auto_sync_seconds']==300
    assert sessions[0] != sessions[1]
    first=sessions[0]
    assert request('POST','/api/visitor/logout',{},first)[0]==200
    assert request('GET','/api/visitor/accounts',headers=first)[0]==401
    assert request('GET','/api/visitor/accounts',headers=sessions[1])[0]==200
finally:
    for credentials in sessions:
        request('POST','/api/visitor/logout',{},credentials)
print('PASS: original 16,376 messages; allowed origin; CORS preflight and errors; two isolated ephemeral sessions, logout and remaining owner.')
