import assert from 'node:assert/strict';
import {browser} from './browser.mjs';
const c=await browser();let issued=false;
try{
 await c.size(375,780,true);await c.go('http://localhost:8766/#accounts');
 await c.wait("byId('visitor-accounts-dialog').open && byId('visitor-account-add').open");
 // Use only an empty temporary session. No address or credential is submitted to the server.
 await c.js("(async()=>{const r=await fetchJSON('/api/visitor/session',new AbortController(),{});const key='ayai-mail:visitor:v1:'+apiOrigin+':'+new URL('.',location.href).pathname;sessionStorage.setItem(key,JSON.stringify({token:r.token,expires:Date.now()+r.expires_in*1000}));})()");issued=true;
 await c.cmd('Page.reload',{ignoreCache:true});await c.wait("byId('visitor-accounts-dialog').open && !byId('visitor-accounts-logout').hidden && visitorMail.token()");
 assert.ok(await c.js("byId('visitor-accounts-list').children.length===0 && document.documentElement.scrollWidth<=innerWidth"));
 await c.click('visitor-accounts-logout');await c.wait("!visitorMail.token() && !byId('visitor-accounts-dialog').open && state.boxes.length>=12");
 assert.ok(!await c.js("Object.keys(sessionStorage).some(k=>k.includes('visitor:v1'))"));
 assert.deepEqual(c.errors,[]);console.log('PASS actual authenticated empty visitor on localhost:8766, reload, mobile and server-confirmed logout; no IMAP account created.');
}finally{
 if(issued){await c.js("(async()=>{if(visitorMail.token())await fetchJSON('/api/visitor/logout',new AbortController(),{});})()").catch(()=>{})}
 await c.close();
}
