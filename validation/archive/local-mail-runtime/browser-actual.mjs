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
await cmd('Page.enable');await cmd('Runtime.enable');await cmd('Page.navigate',{url:'http://localhost:8766/#accounts'});
await poll("document.querySelector('#accounts-dialog')?.open && document.querySelector('#accounts-frequency')?.hidden===false");
await js("document.querySelector('#accounts-close').click()");
await poll("Array.from(document.querySelector('#mobile-box-select').options).some(o=>o.textContent.includes('Ящик Сереги'))");
await js("const box=document.querySelector('#mobile-box-select'); const archive=[...box.options].find(o=>o.textContent.includes('Ящик Сереги')); box.value=archive.value;box.dispatchEvent(new Event('change'))");
await poll("document.getElementById('page-context').textContent.includes('Ящик Сереги') && document.querySelector('#list-box').getAttribute('aria-busy')==='false' && !!document.querySelector('.mail-row')");
await js("document.querySelector('.mail-row').click()");
await poll("document.querySelector('#reader').open && !!document.querySelector('.message-body')");
await js("document.querySelector('#reader-close').click(); document.querySelector('#search-input').value='письма о заказах'; document.getElementById('search-form').requestSubmit()");
await poll("document.querySelector('#answer-panel').hidden===false && document.querySelector('#list-box').getAttribute('aria-busy')==='false' && !!document.querySelector('.mail-row')");
await js("document.querySelector('#compose-open').click()");
await poll("document.getElementById('compose-dialog').open && !!document.getElementById('compose-body')");
await js("const b=document.getElementById('compose-body'); b.value='Черновик проверки';b.dispatchEvent(new Event('input',{bubbles:true})); document.getElementById('compose-close').click(); document.getElementById('compose-open').click()");
await poll("document.getElementById('compose-dialog').open && document.getElementById('compose-body').value==='Черновик проверки'");
await js("const rec=document.getElementById('compose-to');rec.value='reader+test@example.org';rec.dispatchEvent(new Event('input',{bubbles:true}));const sub=document.getElementById('compose-subject');sub.value='Тема & тест';sub.dispatchEvent(new Event('input',{bubbles:true}))");
await poll("document.getElementById('compose-external').getAttribute('href')?.startsWith('mailto:')");
const correct=await js("(()=>{const u=new URL(document.getElementById('compose-external').getAttribute('href'));return decodeURIComponent(u.pathname)==='reader+test@example.org' && u.searchParams.get('subject')==='Тема & тест' && u.searchParams.get('body')==='Черновик проверки'})()");
if(!correct)throw Error('Draft fields missing from client handoff');
if(errors.length)throw Error(errors.join(','));
console.log('PASS: actual archive dropdown, reader, search, accounts, composer, unsent draft. No private data emitted.');
ws.close();
} finally {chrome.kill('SIGTERM');await Promise.race([once(chrome,'exit'),sleep(3000)]);await fs.rm(profile,{recursive:true,force:true,maxRetries:3,retryDelay:200})}
