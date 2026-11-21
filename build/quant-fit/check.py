import json
import time
from pathlib import Path
from urllib.request import urlopen,Request
from pinkprocess.kit_app import analyst_evidence
root=Path('build/quant-fit')
tag='pinkprocess-fit:lfm12b-iq1s'
out=[]
for filename in ['analyst-input.json','analyst-board-window.json']:
    snapshot=json.loads((Path('build/live-kit')/filename).read_text())
    e=analyst_evidence(snapshot)
    facts={k:e[k] for k in ('observed_now','observed_peak','forecast_interpretation')}
    question="What went wrong? Is it still happening, and can we tell if the database caused it?"
    body={'model':tag,'stream':False,'options':{'num_ctx':768,'num_predict':130,'temperature':0,'seed':42},'messages':[{'role':'system','content':'Diagnose a demo connection pool. Use only the measured facts and FPGA forecast. Be concise. State what is unknown.'},{'role':'user','content':json.dumps(facts)+'\n'+question}]}
    result=json.load(urlopen(Request('http://localhost:11434/api/chat',json.dumps(body).encode(),{'Content-Type':'application/json'}),timeout=120))
    row={'model':tag,'capture':filename,'question':question,'facts':facts,'answer':result['message']['content'],'generated_tokens':result.get('eval_count'),'prompt_tokens':result.get('prompt_eval_count'),'stop':result.get('done_reason')}
    out.append(row)
    print(row,flush=True)
(root/'results.json').write_text(json.dumps(out,indent=2)+'\n')
