import {browser} from './browser.mjs';
const c=await browser();
try{
 await c.size(1464,840);await c.go('http://localhost:8766/');
 await c.wait("Array.from(document.querySelectorAll('#mobile-box-select option')).some(o=>o.value==='box-f9410417c091')");
 if(process.argv[2]!=='accounts'){
  await c.js("const sel=document.getElementById('mobile-box-select');sel.value='box-f9410417c091';sel.dispatchEvent(new Event('change'))");
  await c.js("const asOf=document.getElementById('asof-date');asOf.value='2027-12-19';asOf.dispatchEvent(new Event('change'))");
  await c.fill('search-input','Сумма заказа SYN-077f40913b?');await c.submit('search-form');
  await c.wait("!![...document.querySelectorAll('#answer-panel button')].find(b=>b.textContent.includes('Получить ответ'))");
  await c.js("[...document.querySelectorAll('#answer-panel button')].find(b=>b.textContent.includes('Получить ответ')).click()");
  await c.wait("document.querySelector('.answer-prose') && /431[,.]03/.test(document.querySelector('.answer-prose').textContent) && document.querySelector('.source-button')",90000);
  await c.screenshot('.cache/visitor-ui/slides/rag-search.png',{x:230,y:0,width:1234,height:555});
  console.log('Saved actual VK search and citation');
 }
 if(process.argv[2]!=='search'){
  await c.wait("!document.getElementById('visitor-open').hidden");await c.click('visitor-open');await c.wait("document.getElementById('visitor-accounts-dialog').open && document.getElementById('visitor-account-add').open");
  const rect=await c.js("(()=>{const r=document.getElementById('visitor-accounts-dialog').getBoundingClientRect();return {x:Math.floor(r.x)-2,y:Math.floor(r.y)-2,width:Math.ceil(r.width)+4,height:Math.ceil(r.height)+4}})()");
  await c.screenshot('.cache/visitor-ui/slides/connect-own-mail.png',rect);
  console.log('Saved actual empty personal mailbox connection dialog');
 }
 if(c.errors.length)throw Error(c.errors.join(' '));
}finally{await c.close()}
