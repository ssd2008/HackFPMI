const {chromium}=require('/Users/kirill/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs');const path=require('node:path');const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--disable-background-networking']});
 try{
  for(const [folder,port] of [['.cache/export-eval','18768'],['.cache/export-eval-02','18770']]){
   let stage='setup';const p=await browser.newPage({viewport:{width:1180,height:780},locale:'ru-RU'});const base='http://127.0.0.1:'+port;let external=0;
   await p.route('**/*',route=>{if(new URL(route.request().url()).origin!==base){external++;return route.abort()}return route.continue()});
   try{
    const raw=fs.readFileSync(path.join(folder,'corpus/documents.jsonl'),'utf8').split(/\r?\n/).filter(x=>x.trim()).map(JSON.parse);
    const byId=new Map(raw.map(r=>[r.doc_id,r]));const boxes=new Set(raw.map(r=>r.synthetic_user));
    stage='inbox';await p.goto(base+'/');await p.locator('.mail-row').first().waitFor();assert.equal(await p.locator('.box-button').count(),boxes.size);
    for(let i=0;i<boxes.size;i++){
     const b=p.locator('.box-button').nth(i);const mailbox=await b.getAttribute('data-id');
     const inboxResponse=p.waitForResponse(x=>x.url().includes('/api/inbox?')&&x.status()===200);
     await b.click();const inbox=await(await inboxResponse).json();
     const latest=await p.locator('#asof-date').inputValue();
     const eligible=raw.filter(r=>r.synthetic_user===mailbox&&(!r.doc_dt||r.doc_dt<=latest));assert.equal(inbox.total,eligible.length);
     const ids=inbox.items.map(x=>x.id);assert.deepEqual(ids,eligible.sort((a,b)=>((b.doc_dt||'').localeCompare(a.doc_dt||'')||b.doc_id.localeCompare(a.doc_id))).slice(0,25).map(x=>x.doc_id));
     stage='reader';await p.locator('.mail-row').first().click();await p.locator('.message-body').waitFor();
     const toggle=p.locator('.body-toggle');if(await toggle.isVisible())await toggle.click();
     assert.equal(await p.locator('.message-body').textContent(),byId.get(ids[0]).doc);
     await p.keyboard.press('Escape');stage='paging';await p.locator('#next-page').click();
     await p.waitForFunction(()=>document.getElementById('paging-label').textContent.startsWith('26–'));
     await p.locator('#prev-page').click();await p.waitForFunction(()=>document.getElementById('paging-label').textContent.startsWith('1–'));
    }
    stage='empty snapshot / RAG';const earliest=raw.map(x=>x.doc_dt).filter(Boolean).sort()[0];const t=new Date(earliest+'T12:00:00Z');t.setUTCDate(t.getUTCDate()-1);const day=t.toISOString().slice(0,10);const mailbox=[...boxes][0];
    const json=await(await p.request.get(base+'/api/inbox?'+new URLSearchParams({mailbox,date:day}))).json();assert.equal(json.total,0);
    const answer=await(await p.request.post(base+'/api/answer',{data:{mailbox,date:day,query:'Какое письмо доступно на эту дату?'}})).json();
    assert.equal(answer.mode,'no_evidence');assert.equal(answer.sources.length,0);
    stage='scope missing ID';const absent=await p.request.get(base+'/api/mail/never-present-diagnostic?'+new URLSearchParams({mailbox,date:day}));assert.equal(absent.status(),404);
    assert.equal(external,0);console.log('Port',port,'inbox, original message, paging, all mailbox selectors, empty timeline/RAG, nonexistent ID passed.');
   }catch(error){console.error('Private UI check did not pass at',stage,'for port',port,'(details suppressed).');process.exitCode=1}finally{await p.close()}
  }
 }finally{await browser.close()}
})().catch(()=>{console.error('Private browser verification unavailable');process.exitCode=1});
