import math,json
from pathlib import Path
import numpy as np
from lm_reference.flan import Flan
from lm_reference.fabric import WIDTH,ROWS
from lm_reference.flan_prompt import telemetry_prompt
from pinkprocess.kit_evidence import analyst_evidence
p=json.loads(Path('kit/evidence/three-diagnosis-traces-physical.json').read_bytes())
case=p['cases'][2];prompt=telemetry_prompt(p['question'],analyst_evidence(case['started']['snapshot']))
class CountingFabric:
    def __init__(self):
        self.count=0
    def matmat_array(self,w,q,rows):
        assert q.dtype==np.int8 and q.shape[1]%WIDTH==0
        self.count+=int(np.any(q.reshape(len(q),-1,WIDTH),axis=2).sum())*math.ceil(rows/(ROWS-3))
        return q.astype('float32')@np.asarray(w).astype('float32').T
fabric=CountingFabric()
class Profile(Flan):
    def linear(self,name,x):
        result=super().linear(name,x)
        if name=='lm_head':
            self.steps.append(fabric.count)
        return result
r=Profile(Path('build/flan/diagnostic'),fabric)
r.steps=[]
answer=r.generate(prompt)
assert answer==case['result']['answer']
assert r.fabric_projections==case['result']['fabric_projections']
print('input tokens',len(r.tokenizer.encode(prompt)),'decode steps',len(r.steps),'tile runs through first token',r.steps[0],'all',fabric.count,'execution-only floor seconds',[round(x*2*512*16/50_000_000,3) for x in (r.steps[0],fabric.count)])
