import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
export const delay=ms=>new Promise(r=>setTimeout(r,ms));
export async function waitFor(f,ms=12000){const end=Date.now()+ms;while(Date.now()<end){try{const x=await f();if(x)return x}catch{}await delay(50)}throw Error('Timed out: '+String(f).slice(0,120))}
export async function browser(){
 const profile=await fs.mkdtemp(path.join(os.tmpdir(),'private-mail-ui-'));
 const chrome=spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',['--headless=new','--disable-gpu','--disable-background-networking','--disable-sync','--no-first-run','--remote-debugging-address=127.0.0.1','--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{stdio:'ignore'});
 const port=await waitFor(()=>fs.readFile(path.join(profile,'DevToolsActivePort'),'utf8').then(s=>s.split('\n')[0],()=>0));
 const tabs=await(await fetch('http://127.0.0.1:'+port+'/json/list')).json();
 const ws=new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);await once(ws,'open');
 let seq=0;const pending=new Map(),errors=[];
 ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.id){const [ok,no]=pending.get(m.id);pending.delete(m.id);m.error?no(Error(m.error.message)):ok(m.result)}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.exception?.description||m.params.exceptionDetails.text)};
 const cmd=(method,params={})=>new Promise((ok,no)=>{const id=++seq;pending.set(id,[ok,no]);ws.send(JSON.stringify({id,method,params}))});
 const js=async expr=>{const r=await cmd('Runtime.evaluate',{expression:expr,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(r.exceptionDetails.exception?.description||r.exceptionDetails.text);return r.result.value};
 const wait=async (expr,ms)=>{try{return await waitFor(()=>js(expr),ms)}catch{throw Error('UI timeout: '+expr+' / '+JSON.stringify(errors))}};
 const go=async url=>{await cmd('Page.navigate',{url});await wait("document.readyState==='complete' && typeof visitorMail!=='undefined'");};
 const click=id=>js("document.getElementById("+JSON.stringify(id)+").click()");
 const fill=(id,v)=>js("(()=>{const e=document.getElementById("+JSON.stringify(id)+");e.value="+JSON.stringify(v)+";e.dispatchEvent(new Event('input',{bubbles:true}));})()");
 const submit=id=>js("document.getElementById("+JSON.stringify(id)+").requestSubmit()");
 const screenshot=async (file,clip)=>{const {data}=await cmd('Page.captureScreenshot',{format:'png',...(clip?{clip:{...clip,scale:1}}:{})});await fs.writeFile(file,Buffer.from(data,'base64'))};
 const size=async (width,height,mobile=false)=>cmd('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile});
 const close=async()=>{ws.close();chrome.kill('SIGTERM');await Promise.race([once(chrome,'exit'),delay(2000)]);await fs.rm(profile,{recursive:true,force:true,maxRetries:4,retryDelay:100})};
 await cmd('Page.enable');await cmd('Runtime.enable');
 return {js,wait,go,click,fill,submit,screenshot,size,close,errors,cmd};
}
