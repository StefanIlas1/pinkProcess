import sys,time,json
from pathlib import Path
from playwright.sync_api import sync_playwright,expect
url,mode=sys.argv[1:3]
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
 page=browser.new_page(viewport={'width':1100,'height':760})
 errors=[];jobs=[];starts=[];uploads=[]
 page.on('pageerror',lambda e:errors.append(str(e)))
 def watch(r):
  if not r.url.endswith('/api/action'):return
  op=r.request.post_data_json['action']
  if op in ('dataset_ask','ask_begin','dataset_upload','answer_progress'):
   data=r.json()
   if op=='dataset_upload':uploads.append(data)
   if op in ('dataset_ask','ask_begin'):starts.append((op,r.request.post_data_json.get('stage'),data))
   if op=='answer_progress' and data.get('state')=='complete':jobs.append(data)
   if op!='answer_progress' or data.get('state')!='running':print(op,data.get('state'),data.get('error'),flush=True)
 page.on('response',watch);page.goto(url)
 page.wait_for_selector('body:not([data-boot])',timeout=30000)
 if page.locator('#offline').is_visible():page.locator('#find').click()
 expect(page.locator('#connection')).to_have_text('PYNQ-Z2 connected',timeout=30000)
 expect(page.locator('#import-more')).to_be_enabled(timeout=30000)
 if mode=='sample':page.get_by_role('button',name='Connection pool',exact=True).click()
 if mode=='csv':
  with page.expect_file_chooser() as fc:page.locator('#import-more').click()
  fc.value.set_files('kit/examples/pool-recovery.csv')
 if mode in ('sample','csv'):
  expect(page.locator('#capacity-dialog')).to_be_visible(timeout=10000)
  page.locator('#capacity-form button[type=submit]').click()
  expect(page.locator('#notice')).to_have_text('Uploaded to the board.',timeout=30000)
  assert len(uploads)==1 and uploads[0]['ok']
 for i in range(2 if mode=='csv' else 1):
  q='What happened to the queue?' if i==0 else 'How many requests are waiting now?'
  start=time.monotonic()
  for retry in range(90):
   page.locator('#live-question').fill(q);page.locator('#ask-live').click()
   page.wait_for_function("document.querySelector('#notice').dataset.error === 'true' || document.querySelector('.skeleton') !== null",timeout=20000)
   page.wait_for_timeout(1800)
   if 'in progress' not in page.locator('#notice').inner_text():break
   page.wait_for_timeout(2500)
  try:expect(page.locator('.gen-timing')).to_have_count(i+1,timeout=145000)
  except Exception:print('UI error:',page.locator('#notice').text_content(),'errors',errors,flush=True);raise
  expect(page.locator('#import-more')).to_be_enabled(timeout=30000)
  job=jobs[-1];assert job['accelerated_projections']==job['fabric_projections']>0
  op,stage,data=starts[-1];assert (op,stage)==(('ask_begin',None) if mode=='live' else ('dataset_ask','final' if i==0 else 'followup'))
  print(mode,stage,round(time.monotonic()-start,1),job['fabric_projections'],job['answer'],page.locator('.queue-history').last.inner_text(),flush=True)
  if mode=='csv':assert 'cleared to 0' in page.locator('.queue-history').last.inner_text()
  if mode!='live':
   for label in ['Short','Summary','Final','Transcript','Table']:
    page.locator('.vt').get_by_text(label,exact=True).count()
    page.locator('.vt').filter(has_text=label).last.click()
 page.locator('#export-btn').click()
 with page.expect_download() as dl:page.locator('[data-export=json]').click()
 exported=json.loads(Path(dl.value.path()).read_text()); assert jobs[-1]['answer'] in json.dumps(exported,ensure_ascii=False) or jobs[-1]['answer'] in str(exported)
 page.screenshot(path=f'/tmp/pynq-c6-verified-{mode}.png');assert not errors,errors
 print('Export verified',flush=True)
