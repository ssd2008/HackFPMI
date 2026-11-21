import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {once} from 'node:events';

const delay=ms=>new Promise(r=>setTimeout(r,ms));
const db={mode:'local', accounts:[],calls:[],connectFail:true,sendFail:true,seq:0};
const boxes=Array.from({length:12},(_,i)=>({id:'box-'+String(i).padStart(3,'0'),count:2,latest_date:'2029-05-04'}));
const named={id:'a-private-snapshot',kind:'archive',label:'Ящик Сереги',count:1,latest_date:'2025-04-04'};
const root=process.cwd();
const server=http.createServer(async(req,res)=>{
  const u=new URL(req.url,'http://127.0.0.1'), route=u.pathname;
  if(route==='/'||route.startsWith('/static/')){
    const file=path.join(root,route==='/'?'static/index.html':route.slice(1));
    const body=await fs.readFile(file);
    res.writeHead(200,{'Content-Type':({'.html':'text/html; charset=utf-8','.css':'text/css','.js':'text/javascript','.svg':'image/svg+xml'})[path.extname(file)],'Cache-Control':'no-store'});res.end(body);return;
  }
  if(!route.startsWith('/api/')){res.writeHead(404);res.end();return}
  let body='';for await (const chunk of req)body+=chunk;
  const data=body?JSON.parse(body):null;
  const log={route,method:req.method,box:u.searchParams.get('mailbox')||data?.mailbox,day:u.searchParams.get('date')||data?.date,header:req.headers['x-mailsearch-local'],type:req.headers['content-type'],data}; db.calls.push(log);
  const out=(v,status=200)=>{res.writeHead(status,{'Content-Type':'application/json','Cache-Control':'no-store'});res.end(JSON.stringify(v))};
  if (route==='/api/mailboxes') return out({default_mailbox:boxes[0].id,default_date:'2029-05-04',local_mail:db.mode==='local',mailboxes:db.mode==='local'?[...boxes,named,...db.accounts.map(a=>({id:a.id,kind:'live',label:a.address,count:a.count,latest_date:null}))]:boxes});
  if(route==='/api/local/accounts'&&req.method==='GET')return out({accounts:db.accounts,auto_sync_seconds:60});
  if(route==='/api/local/accounts'&&req.method==='POST'){
    if(db.connectFail){db.connectFail=false;return out({detail:'Не принят тестовый пароль приложения.'},401)}
    const acc={id:'live-'+(++db.seq),address:data.address,provider:data.provider,count:0,last_synced:null,status:'idle',pending:0,error:null};db.accounts.push(acc);return out({account:acc});
  }
  if(route.endsWith('/sync')){
    const a=db.accounts.find(a=>route.includes(a.id));a.status='syncing';a.pending=4;a.count+=2;a.last_synced='2026-11-21T10:00:00Z';return out({account:a})
  }
  if(route==='/api/local/send'){const ms=db.slow?900:160;db.slow=false;await delay(ms);if(db.uncertain){db.uncertain=false;return out({detail:'Однозначного подтверждения доставки нет.'},409)}if(db.sendFail){db.sendFail=false;return out({detail:'Временная ошибка отправки.'},502)}return out({status:'sent'})}
  if(route==='/api/reminder-candidates')return out({items:[]});
  if(route==='/api/inbox'||route==='/api/actions')return out({total:0,items:[],offset:0,limit:25});
  if(route==='/api/search')return out({items:[],elapsed_ms:1});
  return out({detail:'Отсутствует'},404);
});
server.listen(0,'127.0.0.1');await once(server,'listening');const origin='http://127.0.0.1:'+server.address().port;
const profile=await fs.mkdtemp(path.join(os.tmpdir(),'mail-ui-local-'));
const chrome=spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',['--headless=new','--disable-gpu','--disable-background-networking','--disable-sync','--no-first-run','--remote-debugging-address=127.0.0.1','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{stdio:'ignore'});
async function until(f,ms=8000){const end=Date.now()+ms;while(Date.now()<end){const x=await f();if(x)return x;await delay(35)}throw Error('Timeout: '+String(f).slice(0,160))}
try {
 const port=await until(()=>fs.readFile(path.join(profile,'DevToolsActivePort'),'utf8').then(s=>s.split('\n')[0],()=>0));
 const tabs=await(await fetch('http://127.0.0.1:'+port+'/json/list')).json();const ws=new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);await once(ws,'open');let seq=0;const pending=new Map(),errors=[];
 ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const [ok,no]=pending.get(m.id);pending.delete(m.id);m.error?no(Error(m.error.message)):ok(m.result)}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.exception?.description||m.params.exceptionDetails.text)};
 const cmd=(method,params={})=>new Promise((ok,no)=>{const id=++seq;pending.set(id,[ok,no]);ws.send(JSON.stringify({id,method,params}))});
 const js=async expr=>{const r=await cmd('Runtime.evaluate',{expression:expr,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(r.exceptionDetails.exception?.description||r.exceptionDetails.text);return r.result.value};
 const wait=async expr=>{try{return await until(()=>js(expr))}catch{throw Error('UI timeout: '+expr+' / '+JSON.stringify(errors))}};
 const go=async suffix=>{await cmd('Page.navigate',{url:origin+suffix});await wait("document.readyState==='complete' && typeof localMail!=='undefined'");};
 const click=id=>js("document.getElementById("+JSON.stringify(id)+").click()");
 const fill=(id,v)=>js("(()=>{const e=document.getElementById("+JSON.stringify(id)+");e.value="+JSON.stringify(v)+";e.dispatchEvent(new Event('input',{bubbles:true}));})()");
 const submit=id=>js("document.getElementById("+JSON.stringify(id)+").requestSubmit()");
 const val=id=>js("document.getElementById("+JSON.stringify(id)+").value");
 const aux=()=>db.calls.filter(c=>c.route.startsWith('/api/local/')).length;
 const sent=()=>db.calls.filter(c=>c.route==='/api/local/send');
 const open=id=>wait('document.getElementById('+JSON.stringify(id)+').open');
 const screenshot=async name=>{const {data}=await cmd('Page.captureScreenshot',{format:'png'});await fs.writeFile(path.join(root,'.cache/local-mail-ui',name),Buffer.from(data,'base64'))};
 await cmd('Page.enable');await cmd('Runtime.enable');
 await cmd('Emulation.setDeviceMetricsOverride',{width:1220,height:810,deviceScaleFactor:1,mobile:false});
 await go('/#accounts');await open('accounts-dialog');await wait("document.getElementById('account-add').open && document.getElementById('accounts-list').textContent.includes('Пока')");
 assert.equal(await js("document.querySelectorAll('#mobile-box-select option').length"),13);
 assert.deepEqual(await js("Array.from(document.querySelectorAll('#mobile-box-select option')).slice(1).map(x=>x.textContent)"),Array.from({length:12},(_,i)=>'Ящик '+String(i+1).padStart(2,'0')));
 assert.equal(await js("document.getElementById('account-password-link').href"),'https://myaccount.google.com/apppasswords');assert.ok(await js("[...document.querySelectorAll('#local-controls > *')].every(b=>{const rect=b.getBoundingClientRect(), text=[...b.childNodes].find(n=>n.nodeType===3 && n.textContent.trim()),range=document.createRange();range.selectNode(text);return range.getBoundingClientRect().right<rect.right && b.querySelector('svg').getBoundingClientRect().left>rect.left})"));await screenshot('accounts-desktop.png');await click('accounts-close');await wait("!location.hash");
 await cmd('Emulation.setDeviceMetricsOverride',{width:375,height:810,deviceScaleFactor:1,mobile:true});
 assert.ok(await js("document.documentElement.scrollWidth<=innerWidth"));
 await js("const mobile=document.getElementById('mobile-box-select');mobile.value='a-private-snapshot';mobile.dispatchEvent(new Event('change'))");
 await wait("document.getElementById('page-context').textContent.includes('Ящик Сереги')");
 assert.equal(await val('asof-date'),'2025-04-04');
 assert.ok(db.calls.some(c=>c.route==='/api/inbox'&&c.box===named.id));
 await fill('search-input','пример');await submit('search-form');await wait("document.getElementById('page-heading').textContent.includes('Поиск')");
 await fill('search-input','ещё не отправленный запрос');
 await click('compose-open');await open('compose-dialog');assert.equal(await js("document.getElementById('compose-send').disabled"),true);
 assert.equal(await js("document.getElementById('compose-external').hasAttribute('href')"),false);
 await fill('compose-to','alpha+tag@example.invalid, beta@example.invalid');await fill('compose-subject','Статус & следующий? 100% + #');
 await fill('compose-body','Первая строка\n&вторая + тема? #фрагмент\nКириллица');
 const raw=await js("document.getElementById('compose-external').getAttribute('href')");
 const mail=new URL(raw);assert.equal(mail.protocol,'mailto:');assert.equal(mail.pathname,'alpha%2Btag@example.invalid,beta@example.invalid');
 assert.deepEqual([...mail.searchParams],[['subject','Статус & следующий? 100% + #'],['body','Первая строка\r\n&вторая + тема? #фрагмент\r\nКириллица']]);
 assert.equal(sent().length,0);assert.equal(mail.hash,'');assert.ok(!raw.includes('account_id')&&!raw.includes('password'));
 await fill('compose-to','неадрес');assert.equal(await js("document.getElementById('compose-external').hasAttribute('href')"),false);
 await fill('compose-to','alpha@example.invalid, beta@example.invalid');await fill('compose-subject','Тест');await fill('compose-body','Это тестовый черновик.');
 await screenshot('compose-mobile.png');await click('compose-close');await click('compose-open');assert.equal(await val('compose-body'),'Это тестовый черновик.');
 await click('compose-connect');await open('accounts-dialog');await fill('account-address','owner@example.invalid');const password='fake-unique-application-password';await fill('account-password',password);
 await submit('account-form');await wait("document.getElementById('accounts-feedback').textContent.includes('Не принят')");assert.equal(await val('account-password'),'');
 assert.ok(!await js("JSON.stringify([localStorage,sessionStorage]).includes("+JSON.stringify(password)+")"));
 await fill('account-password',password);await submit('account-form');await wait("!document.getElementById('accounts-dialog').open && document.getElementById('compose-from').value==='live-1'");
 assert.equal(await val('compose-body'),'Это тестовый черновик.');assert.equal(await val('search-input'),'ещё не отправленный запрос');
 assert.equal(await val('mobile-box-select'),named.id);assert.equal(await js("Array.from(document.getElementById('compose-from').options).length"),1);
 await js("document.getElementById('compose-form').requestSubmit();document.getElementById('compose-form').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}))");
 await wait("document.getElementById('compose-feedback').textContent.includes('Временная')");assert.equal(sent().length,1);
 const first=sent()[0].data;assert.deepEqual(first.to,['alpha@example.invalid','beta@example.invalid']);assert.equal(first.account_id,'live-1');
 assert.match(first.request_id,/^[a-f\d]{8}-[a-f\d]{4}-4[a-f\d]{3}-[89ab][a-f\d]{3}-[a-f\d]{12}$/i);
 await cmd('Page.reload',{ignoreCache:true});await wait("document.getElementById('local-controls') && !document.getElementById('local-controls').hidden && document.querySelectorAll('#mobile-box-select option').length===14");
 await click('compose-open');await open('compose-dialog');await wait("document.getElementById('compose-from').value==='live-1'");
 assert.equal(await val('compose-body'),'Это тестовый черновик.');await submit('compose-form');await wait("!document.getElementById('compose-dialog').open && document.getElementById('notifications').textContent.includes('Письмо отправлено')");
 assert.equal(sent().length,2);assert.equal(sent()[1].data.request_id,first.request_id);assert.equal(await js("sessionStorage.getItem('ayai-mail:local:draft:1')"),null);
 await click('compose-open');await fill('compose-to','alpha@example.invalid');await submit('compose-form');await wait("document.getElementById('compose-feedback').textContent.includes('Добавьте тему')");assert.equal(sent().length,2);
 db.sendFail=true;await fill('compose-body','Только текст');await submit('compose-form');await wait("document.getElementById('compose-feedback').textContent.includes('Временная')");await fill('compose-subject','Изменённое содержание');
 await submit('compose-form');await wait("!document.getElementById('compose-dialog').open");assert.notEqual(sent()[2].data.request_id,sent()[3].data.request_id);
 await click('compose-open');await fill('compose-to','alpha@example.invalid');await fill('compose-subject','Только тема');db.uncertain=true;await submit('compose-form');
 await wait("!document.getElementById('compose-review').hidden && document.getElementById('compose-feedback').textContent.includes('подтверждения')");
 assert.equal(await js("document.getElementById('compose-send').disabled && !document.getElementById('compose-external').hasAttribute('href')"),true);
 const ambiguous=sent().at(-1), ambiguousCount=sent().length;await submit('compose-form');await delay(250);assert.equal(sent().length,ambiguousCount);
 await cmd('Page.reload',{ignoreCache:true});await wait("document.getElementById('local-controls') && !document.getElementById('local-controls').hidden && document.querySelectorAll('#mobile-box-select option').length===14");
 await click('compose-open');await open('compose-dialog');await wait("!document.getElementById('compose-review').hidden && document.getElementById('compose-send').disabled && document.getElementById('compose-review-hint').textContent.includes('Отправленные')");
 assert.equal(await val('compose-subject'),'Только тема');assert.equal(await js("document.getElementById('compose-external').hasAttribute('href')"),false);
 await click('compose-checked');assert.ok(await js("!document.getElementById('compose-send').disabled && document.getElementById('compose-external').href.startsWith('mailto:')"));
 await submit('compose-form');await wait("!document.getElementById('compose-dialog').open");assert.equal(sent().at(-1).data.request_id,ambiguous.data.request_id);
 await click('compose-open');await fill('compose-to','alpha@example.invalid');await fill('compose-body','Обновление вкладки при отправке');
 db.slow=true;const initialSent=sent().length;await submit('compose-form');await until(()=>sent().length>initialSent);const interrupted=sent().at(-1);
 await cmd('Page.reload',{ignoreCache:true});await wait("document.getElementById('local-controls') && !document.getElementById('local-controls').hidden && document.querySelectorAll('#mobile-box-select option').length===14");
 await click('compose-open');await open('compose-dialog');await wait("!document.getElementById('compose-review').hidden && document.getElementById('compose-send').disabled");
 assert.equal(await js("document.getElementById('compose-external').hasAttribute('href')"),false);await click('compose-checked');
 await submit('compose-form');await wait("!document.getElementById('compose-dialog').open");assert.equal(sent().at(-1).data.request_id,interrupted.data.request_id);
 await js("const m=document.getElementById('mobile-box-select');m.value='live-1';m.dispatchEvent(new Event('change'))");assert.equal(await val('asof-date'),await js('localMail.today()'));
 await fill('search-input','сохранить выдачу');await submit('search-form');await wait("document.getElementById('page-heading').textContent.includes('Поиск')");
 await click('accounts-open');await open('accounts-dialog');await js("document.querySelector('[data-sync-account]').click()");await wait("document.getElementById('accounts-list').textContent.includes('В очереди на загрузку 4 письма')");
 assert.equal(await val('mobile-box-select'),'live-1');assert.equal(await val('search-input'),'сохранить выдачу');
 db.accounts[0].status='error';db.accounts[0].error='Тестовая причина синхронизации';await click('accounts-retry');await wait("document.getElementById('accounts-list').textContent.includes('Тестовая причина')");
 await js("document.getElementById('account-add').open=true");await js("const prov=document.getElementById('account-provider');prov.value='yandex';prov.dispatchEvent(new Event('change'))");assert.equal(await js("document.getElementById('account-password-link').href"),'https://passport.yandex.ru/security/app-passwords');await fill('account-address','second@example.invalid');await fill('account-password','fake-yandex-key');await submit('account-form');await wait("document.querySelectorAll('#accounts-list .account-card').length===2");
 await screenshot('accounts-mobile.png');assert.ok(await js("document.documentElement.scrollWidth<=innerWidth && document.getElementById('accounts-dialog').scrollWidth<=document.getElementById('accounts-dialog').clientWidth+1"));
 assert.ok(db.calls.filter(c=>c.route.startsWith('/api/local/')&&c.method==='POST').every(c=>c.header==='1'&&c.type==='application/json'));
 assert.ok(db.calls.filter(c=>!c.route.startsWith('/api/local/')||c.method==='GET').every(c=>!c.header));
 const writes=db.calls.filter(c=>c.route==='/api/local/accounts'&&c.method==='POST');assert.deepEqual(writes.map(c=>c.data.provider),['gmail','gmail','yandex']);
 let n=aux();db.mode='public';await go('/?view=public#accounts');await wait("document.querySelectorAll('#mobile-box-select option').length===12");
 assert.ok(await js("document.getElementById('local-controls').hidden && !document.getElementById('accounts-dialog').open"));await delay(200);assert.equal(aux(),n);
 await go('/?demo=1#accounts');await wait("!document.getElementById('demo-label').hidden");assert.ok(await js("document.getElementById('local-controls').hidden && !document.getElementById('accounts-dialog').open"));await delay(150);assert.equal(aux(),n);
 assert.deepEqual(errors,[]);console.log('PASS desktop and mobile local UI, labels and as-of date, live handles, both providers, pending and errors, draft persistence and idempotent resend, header confinement, public/demo isolation. Encoded mailto and ambiguity safeguards passed. All provider routes mocked.');
 ws.close();
} finally {chrome.kill('SIGTERM');await Promise.race([once(chrome,'exit'),delay(2500)]);await fs.rm(profile,{recursive:true,force:true,maxRetries:3,retryDelay:150});server.close()}
