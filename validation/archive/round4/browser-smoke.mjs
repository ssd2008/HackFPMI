import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {once} from 'node:events';

const profile=await fs.mkdtemp(path.join(os.tmpdir(),'mail-search-smoke-'));
const chrome=spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',['--headless=new','--disable-gpu','--disable-background-networking','--disable-sync','--no-first-run','--remote-debugging-address=127.0.0.1','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{stdio:'ignore'});
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function until(f,ms=30000){const limit=Date.now()+ms; while(Date.now()<limit){const value=await f();if(value)return value;await sleep(100)}throw Error('test condition timed out: '+String(f).slice(0,100))}
try{
const port=await until(async()=>fs.readFile(path.join(profile,'DevToolsActivePort'),'utf8').then(s=>s.split('\n')[0],()=>0));
const tabs=await (await fetch('http://127.0.0.1:'+port+'/json/list')).json();
const ws=new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);await once(ws,'open');
const outstanding=new Map(),errors=[];let seq=0;
ws.onmessage=event=>{const msg=JSON.parse(event.data);if(msg.id){const p=outstanding.get(msg.id);if(p){outstanding.delete(msg.id);msg.error?p[1](Error(msg.error.message)):p[0](msg.result)}}else if(msg.method==='Runtime.exceptionThrown')errors.push(msg.params.exceptionDetails.text)};
const cmd=(method,params={})=>new Promise((ok,no)=>{const id=++seq;outstanding.set(id,[ok,no]);ws.send(JSON.stringify({id,method,params}))});
const js=async(expression)=>{const o=await cmd('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(o.exceptionDetails)throw Error(o.exceptionDetails.exception?.description||'evaluation failed');return o.result.value};
const poll=expression=>until(()=>js(expression));
await cmd('Page.enable');await cmd('Runtime.enable');await cmd('Page.navigate',{url:'http://localhost:8766/'});
await poll("document.querySelectorAll('#mobile-box-select option').length===12 && document.querySelectorAll('.mail-row').length>0");
await js("document.querySelector('#nav-actions').click()");await poll("document.querySelector('#page-heading').innerText.includes('действ') && document.querySelector('#list-box').getAttribute('aria-busy')==='false'");
await js("document.querySelector('#nav-inbox').click(); const b=document.querySelector('#mobile-box-select');b.value='box-f9410417c091';b.dispatchEvent(new Event('change'))");
await poll("document.querySelector('#mobile-box-select').value==='box-f9410417c091' && document.querySelector('#list-box').getAttribute('aria-busy')==='false'");
await js("const d=document.querySelector('#asof-date');d.value='2027-12-19';d.dispatchEvent(new Event('change'))");
await poll("document.querySelector('#list-box').getAttribute('aria-busy')==='false'");
await js("document.querySelector('#search-input').value='Какова сумма заказа SYN-077f40913b?';document.querySelector('#search-form').requestSubmit()");
await poll("!document.querySelector('#answer-panel').hidden && document.querySelector('.mail-row') && document.querySelector('#list-box').getAttribute('aria-busy')==='false'");
await js("document.querySelector('#answer-panel button').click()");
await poll("document.querySelector('#answer-panel .source-button') && /431[.,]03/.test(document.querySelector('#answer-panel').textContent)");
await js("document.querySelector('.source-button').click()");
await poll("document.querySelector('#reader').open && /431[.,]03/.test(document.querySelector('#reader-content').textContent)");
await js("document.querySelector('#reader-close').click();document.querySelector('#search-input').value='«несусветноедивонепохожее»';document.querySelector('#search-form').requestSubmit()");
await poll("document.querySelector('#list-box').getAttribute('aria-busy')==='false' && [...document.querySelectorAll('#list-box button')].some(x=>x.textContent.includes('без кавычек'))");
await js("document.querySelector('#list-box button').click()");
await poll("!document.querySelector('#search-input').value.includes('«') && document.querySelector('#list-box').getAttribute('aria-busy')==='false'");
if(errors.length) throw Error(errors.join(','));
console.log('PASS: 12 mailboxes, inbox and action views, date and mailbox changes, search, factual RAG response original cited message and retrying an empty quotation without quotes; no browser exceptions');
ws.close();
} finally {chrome.kill('SIGTERM');await Promise.race([once(chrome,'exit'),sleep(3000)]);await fs.rm(profile,{recursive:true,force:true,maxRetries:3,retryDelay:200})}
