import {browser} from './browser.mjs';
import assert from 'node:assert/strict';
const [pageUrl,expectedRevision]=process.argv.slice(2);
const c=await browser();
try {
  await c.size(1380,800);await c.go(pageUrl+'?v='+expectedRevision);
  await c.wait("document.getElementById('visitor-open') && !document.getElementById('visitor-open').hidden",45000);
  await c.wait("document.querySelectorAll('#mobile-box-select option').length===12");
  const version = await c.js("fetch(new URL('version.txt?r='+Date.now(),location.href),{cache:'no-store'}).then(r=>r.text()).then(t=>t.trim())");
  assert.equal(version,expectedRevision);
  await c.click('visitor-open');
  await c.wait("document.querySelector('#visitor-accounts-dialog[open]') && document.getElementById('visitor-account-add').open");
  const labels=await c.js("document.querySelector('#visitor-accounts-dialog').innerText");
  for(const word of ['Gmail','Яндекс','500','час']) assert.ok(labels.includes(word),word);
  const stats=await c.js(`(async()=>{
    const api=document.querySelector('meta[name=mailsearch-api-origin]').content;
    async function request(path,token,body){
      const headers={};
      if(body){headers['Content-Type']='application/json';headers['X-Mailsearch-Visitor']='1';}
      if(token)headers.Authorization='Bearer '+token;
      return fetch(api+'/'+path,{method:body?'POST':'GET',credentials:'omit',headers,body:body?JSON.stringify(body):undefined,cache:'no-store',redirect:'error'});
    }
    const opened=await request('api/visitor/session',null,{}); if(!opened.ok) throw Error('session '+opened.status);
    const {token,expires_in}=await opened.json();
    const mine=await request('api/visitor/accounts',token);
    const accounts=(await mine.json()).accounts;
    const out=await request('api/visitor/logout',token,{});
    const after=await request('api/visitor/accounts',token);
    return {scheme:new URL(api).protocol,session:opened.status,ttl:expires_in,accounts:accounts.length,status:mine.status,logout:out.status,after:after.status,cache:mine.headers.get('cache-control')};
  })()`);
  assert.equal(stats.scheme,'https:');assert.equal(stats.ttl,3600);assert.equal(stats.status,200);assert.equal(stats.accounts,0);assert.ok([200,204].includes(stats.logout));assert.equal(stats.after,401);assert.match(stats.cache,/no-store/);
  await c.click('visitor-accounts-close');
  await c.js("const choice=document.querySelector('#mobile-box-select');choice.value='box-f9410417c091';choice.dispatchEvent(new Event('change'))");
  await c.js("const calendar=document.getElementById('asof-date');calendar.value='2027-12-19';calendar.dispatchEvent(new Event('change'))");
  await c.fill('search-input','SYN-077f40913b');await c.submit('search-form');
  await c.wait("document.querySelector('.mail-row') && /SYN-077f40913b/i.test(document.querySelector('.mail-row').innerText)",30000);
  await c.js("document.querySelector('.mail-row').click()");
  await c.wait("document.querySelector('#reader[open],#reader-dialog[open],dialog[open]') && /431[.,]03/.test(document.querySelector('dialog[open]').innerText)",20000);
  assert.deepEqual(c.errors,[]);
  console.log(JSON.stringify({revision:version,visibleMailboxes:12,providers:2,sourceOpened:true,scope:stats}));
} finally {await c.close();}
