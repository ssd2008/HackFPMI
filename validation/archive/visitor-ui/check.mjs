import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import {once} from 'node:events';
import {browser,waitFor,delay} from './browser.mjs';

const address='fixture@example.invalid', ownerTag='PRIVATE_VIEW_TEST', password='FAKE-IMAP-APP-PASSWORD';
const synth=Array.from({length:12},(_,n)=>({id:'box-'+String(n).padStart(2,'0'),count:1,latest_date:'2027-12-01'}));
const log=[], sessions=new Map();let counter=0, failPassword=true;
const listen=async fn=>{const s=http.createServer(fn);s.listen(0,'127.0.0.1');await once(s,'listening');return s};
const url=s=>'http://127.0.0.1:'+s.address().port;
const api=await listen(async(req,res)=>{
 res.setHeader('Access-Control-Allow-Origin',url(web));res.setHeader('Access-Control-Allow-Headers','Authorization, Content-Type, X-Mailsearch-Visitor');res.setHeader('Access-Control-Allow-Methods','GET, POST, OPTIONS');
 if(req.method==='OPTIONS'){res.writeHead(200);res.end();return}
 const u=new URL(req.url,url(api));let body='';for await(const b of req)body+=b;const obj=body?JSON.parse(body):{};
 const token=req.headers.authorization?.replace('Bearer ','') || '';
 const user=sessions.get(token);
 log.push({path:u.pathname,method:req.method,token,marker:req.headers['x-mailsearch-visitor'],box:u.searchParams.get('mailbox')||obj.mailbox});
 const out=(code,d)=>{res.writeHead(code,{'Content-Type':'application/json','Cache-Control':'no-store'});res.end(JSON.stringify(d))};
 if(token&&!user)return out(401,{detail:'Сессия завершена'});
 if(u.pathname==='/api/visitor/session'){assert.equal(token,'');const t='test-token-'+(++counter);sessions.set(t,{accounts:[]});return out(200,{token:t,expires_in:1200})}
 if(u.pathname.startsWith('/api/visitor/')&&!user)return out(401,{detail:'Войдите в сессию'});
 if(u.pathname==='/api/visitor/accounts'&&req.method==='GET')return out(200,{accounts:user.accounts,auto_sync_seconds:60});
 if(u.pathname==='/api/visitor/accounts'&&req.method==='POST'){
  assert.equal(obj.password,password);
  if(failPassword){failPassword=false;return out(403,{detail:'Не подошёл пароль приложения.'})}
  const a={id:'box-private-'+token,provider:obj.provider,address:obj.address,count:1,last_synced:new Date().toISOString(),status:'idle',pending:0,sync_limit:500,total_available:750};user.accounts=[a];return out(200,{account:a})
 }
 if(u.pathname==='/api/visitor/logout'){sessions.delete(token);return out(200,{status:'ok'})}
 if(u.pathname.endsWith('/sync')){user.accounts[0].count+=1;user.accounts[0].pending=3;return out(200,{account:user.accounts[0]})}
 if(u.pathname==='/api/mailboxes')return out(200,{mailboxes:[...synth,...(user?.accounts||[]).map(a=>({id:a.id,kind:'live',label:a.address,count:a.count,latest_date:null}))],default_mailbox:synth[0].id,default_date:'2027-12-01',visitor_mail:true,local_mail:false});
 const privateBox=(u.searchParams.get('mailbox')||obj.mailbox||'').startsWith('box-private');
 if(privateBox&&(!user||user.accounts[0]?.id!==(u.searchParams.get('mailbox')||obj.mailbox)))return out(404,{detail:'Нет ящика'});
 const entry={id:privateBox?'live-message':'demo-doc',sender:'fixture',date:'2026-11-20',subject:privateBox?ownerTag:'Synthetic subject',snippet:privateBox?ownerTag:'fixture',body:privateBox?ownerTag:'Demo'};
 if(u.pathname==='/api/inbox'||u.pathname==='/api/actions')return out(200,{total:1,items:[entry],offset:0,limit:25});
 if(u.pathname==='/api/reminder-candidates')return out(200,{items:[]});
 if(u.pathname.startsWith('/api/mail/'))return out(200,{...entry,truncated:privateBox});
 if(u.pathname==='/api/search')return out(200,{items:[entry],elapsed_ms:2});
 if(u.pathname==='/api/answer')return out(200,{answer:entry.subject+' [1]',mode:'extractive',sources:[entry]});
 if(u.pathname==='/api/assistant')return out(200,{kind:'search',items:[entry]});
 return out(404,{detail:'Неизвестный маршрут'});
});
const siteCalls=[];
const web=await listen(async(req,res)=>{
 const u=new URL(req.url,'http://127.0.0.1');siteCalls.push({path:u.pathname,auth:req.headers.authorization});
 if(u.pathname==='/'||u.pathname.startsWith('/static/')){
  let p=u.pathname==='/'?'static/index.html':u.pathname.slice(1);
  try{let body=await fs.readFile(p);
   if(u.pathname==='/')body=body.toString().replace('<meta name="mailsearch-api-origin" content="">','<meta name="mailsearch-api-origin" content="'+url(api)+'" data-external>');
   res.writeHead(200,{'Content-Type':({'.js':'text/javascript','.css':'text/css','.svg':'image/svg+xml','.html':'text/html; charset=utf-8'})[path.extname(p)]});res.end(body)
  }catch{res.writeHead(404);res.end()}
 }else{res.writeHead(404);res.end()}
});
let a,b;
try{
 [a,b]=await Promise.all([browser(),browser()]);await Promise.all([a.size(1140,810),b.size(350,750,true)]);
 await Promise.all([a.go(url(web)+'/'),b.go(url(web)+'/')]);
 for(const c of [a,b])await c.wait("!byId('visitor-open').hidden && state.boxes.length===12");
 assert.equal(counter,0);assert.equal(log.filter(r=>r.path.startsWith('/api/visitor/')).length,0);
 await a.click('visitor-open');await a.wait("byId('visitor-accounts-dialog').open");
 await a.js("byId('visitor-account-form').querySelector('[value=yandex]').click()");
 assert.match(await a.js("byId('visitor-account-password-link').href"),/passport\.yandex\.ru\/security\/app-passwords/);
 await a.fill('visitor-account-address',address);await a.fill('visitor-account-password',password);await a.submit('visitor-account-form');
 await a.wait("byId('visitor-accounts-feedback').textContent.includes('Не подошёл')");
 assert.equal(await a.js("byId('visitor-account-password').value"),'');assert.ok(!await a.js('JSON.stringify([sessionStorage,localStorage]).includes('+JSON.stringify(password)+')'));
 assert.equal(counter,1);assert.ok(await a.js("Boolean(visitorMail.token())"));
 await a.fill('visitor-account-password',password);await a.submit('visitor-account-form');
 await a.wait("document.querySelector('.visitor-account-actions') && state.boxes.length===13");
 assert.equal(await a.js("byId('mobile-box-select').options[0].textContent"),address);
 assert.match(await a.js("byId('visitor-accounts-list').textContent"),/750 писем/);
 await a.js("document.querySelector('.visitor-account-actions .text-action').click()");await a.wait("state.mailbox.startsWith('box-private') && byId('list-box').textContent.includes('PRIVATE_VIEW_TEST')");
 assert.equal(await a.js("state.date===localMail.today()"),true);
 await a.js("document.querySelector('.mail-row').click()");await a.wait("byId('reader-content').textContent.includes('загружено частично')");
 await a.js("document.querySelector('.mark-action').click()");
 await a.wait("document.querySelector('.mark-action').getAttribute('aria-pressed')==='true'");
 assert.ok(!await a.js('JSON.stringify([localStorage,sessionStorage]).includes('+JSON.stringify(ownerTag)+')'));
 await a.click('reader-close');await a.click('nav-actions');await a.wait("byId('list-box').textContent.includes('PRIVATE_VIEW_TEST')");
 const checks=log.filter(r=>r.path==='/api/actions'&&r.box?.startsWith('box-private'));assert.ok(checks.some(r=>r.token&&r.marker==='1'));
 await a.fill('search-input','личный тест');await a.submit('search-form');await a.wait("document.querySelector('#answer-panel button')&&state.searchResults.length===1");
 await a.js("document.querySelector('#answer-panel button').click()");await a.wait("document.querySelector('.answer-prose')?.textContent.includes('PRIVATE_VIEW_TEST')");
 await a.click('assistant-open');await a.fill('assistant-query','личный вопрос');await a.submit('assistant-form');await a.wait("byId('assistant-results').textContent.includes('PRIVATE_VIEW_TEST')");
 await a.click('assistant-close');await a.click('visitor-open');await a.wait("byId('visitor-accounts-dialog').open");await a.js("document.querySelector('[data-sync-id]').click()");
 await a.wait("byId('visitor-accounts-list').textContent.includes('3 письма')");await a.click('visitor-accounts-close');
 assert.equal(await a.js("state.query"),'личный тест');assert.equal(await b.js("state.boxes.length"),12);assert.equal(await b.js("visitorMail.token()"),'');
 const old=await a.js('visitorMail.token()');await a.cmd('Page.reload',{ignoreCache:true});await a.wait("state.boxes.length===13&&!byId('visitor-open').hidden");assert.equal(await a.js('visitorMail.token()'),old);
 assert.ok(!await a.js("document.body.textContent.includes('PRIVATE_VIEW_TEST') || byId('assistant-query').value"));
 await a.js("openMailbox(state.boxes[0].id)");await a.wait("byId('list-box').textContent.includes('PRIVATE_VIEW_TEST')");
 await a.js("document.querySelector('.mail-row').click()");await a.wait("document.querySelector('.mark-action')");assert.equal(await a.js("document.querySelector('.mark-action').getAttribute('aria-pressed')"),'false');await a.click('reader-close');
 await a.click('visitor-open');await a.wait("byId('visitor-accounts-dialog').open");await a.click('visitor-accounts-logout');
 await a.wait("state.boxes.length===12 && !visitorMail.token() && !byId('visitor-accounts-dialog').open");
 assert.ok(!await a.js("document.body.textContent.includes('PRIVATE_VIEW_TEST') || document.body.textContent.includes('fixture@example.invalid') || JSON.stringify(sessionStorage).includes('test-token-') || byId('search-input').value || byId('assistant-query').value"));
 // Explicitly expire a different browser session while private data is on screen.
 await b.click('visitor-open');await b.fill('visitor-account-address','other@example.invalid');await b.fill('visitor-account-password',password);await b.submit('visitor-account-form');await b.wait("state.boxes.length===13 && document.querySelector('.visitor-account-actions')");
 await b.js("document.querySelector('.visitor-account-actions .text-action').click()");await b.wait("byId('list-box').textContent.includes('PRIVATE_VIEW_TEST')");
 const t2=await b.js('visitorMail.token()');assert.notEqual(t2,old);sessions.delete(t2);
 await b.fill('search-input','истёкшая сессия');await b.submit('search-form');await b.wait("!visitorMail.token() && state.boxes.length===12");
 assert.ok(!await b.js("document.body.textContent.includes('PRIVATE_VIEW_TEST') || document.body.textContent.includes('other@example.invalid') || JSON.stringify(sessionStorage).includes('test-token-') || byId('search-input').value"));
 assert.ok(await b.js("document.documentElement.scrollWidth<=innerWidth"));
 for(const endpoint of ['/api/inbox','/api/mail/live-message','/api/search','/api/answer','/api/actions','/api/assistant','/api/reminder-candidates'])assert.ok(log.some(r=>r.path===endpoint&&r.box?.startsWith('box-private')&&r.token),endpoint);
 assert.ok(log.filter(r=>r.path.startsWith('/api/visitor/')&&r.method==='POST').every(r=>r.marker==='1'));assert.ok(!log.some(r=>r.path.startsWith('/api/local/')));
 assert.ok(!siteCalls.some(c=>c.auth||c.path.startsWith('/api/')));
 try{await a.js("fetchJSON('https://example.invalid/api/search',new AbortController(),{})");assert.fail('foreign request permitted')}catch(e){assert.match(e.message,/Некорректный/)}
 const n=log.length;await a.go(url(web)+'/?demo=1#accounts');await a.wait("state.demo && state.boxes.length===2");
 assert.ok(await a.js("byId('visitor-open').hidden && !byId('visitor-accounts-dialog').open"));await delay(150);assert.equal(log.length,n);
 assert.deepEqual(a.errors,[]);assert.deepEqual(b.errors,[]);
 console.log('PASS two isolated Chrome browsers, separate Pages/API origins, no owner requests, explicit password validation, both flags and providers, inbox/search/read/RAG/actions/assistant bearer, no credential or private storage, 401/logout scrubbing, reloaded session/cleared marks, manual sync, mobile and static offline.');
}finally{await Promise.all([a?.close(),b?.close()]);web.close();api.close();}
