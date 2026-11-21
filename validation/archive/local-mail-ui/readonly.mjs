import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {once} from 'node:events';

const delay=ms=>new Promise(r=>setTimeout(r,ms));
const root=process.cwd(); const origin='http://127.0.0.1:8766';
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
 await cmd('Network.enable');await cmd('Network.setBlockedURLs',{urls:['*/api/local/send','*/api/local/accounts/*/sync']});
 await go('/#accounts');await open('accounts-dialog');await wait("document.getElementById('accounts-list').textContent.includes('Пока нет')");
 assert.equal(await js("document.getElementById('local-controls').hidden"),false);
 assert.equal(await js("Array.from(document.querySelectorAll('#mailboxes button')).filter(b=>b.textContent.includes('Ящик Сереги')).length"),1);
 const count=await js("Array.from(document.getElementById('mobile-box-select').options).filter(opt=>/^Ящик [0-9]{2}$/.test(opt.textContent)).length");assert.equal(count,12);
 await click('accounts-close');await cmd('Emulation.setDeviceMetricsOverride',{width:320,height:700,deviceScaleFactor:1,mobile:true});
 await js("(()=>{const sel=document.getElementById('mobile-box-select');sel.value=[...sel.options].find(o=>o.textContent==='Ящик Сереги').value;sel.dispatchEvent(new Event('change'));})()");
 await wait("document.getElementById('page-context').textContent.includes('Ящик Сереги')");
 assert.ok(await js("state.date===getBox(state.mailbox).latest_date"));await click('compose-open');await open('compose-dialog');
 assert.ok(await js("document.getElementById('compose-send').disabled && !document.getElementById('compose-account-note').hidden && document.getElementById('compose-dialog').clientWidth<=innerWidth && document.documentElement.scrollWidth<=innerWidth"));
 assert.deepEqual(errors,[]);console.log('PASS actual localhost read-only at desktop and 320px: local accounts settings, archived mailbox date, all 12 VK, empty sender list, no sends or credentials.');
 ws.close();
} finally {chrome.kill('SIGTERM');await Promise.race([once(chrome,'exit'),delay(2500)]);await fs.rm(profile,{recursive:true,force:true,maxRetries:3,retryDelay:150})}
