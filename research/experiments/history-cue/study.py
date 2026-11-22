"""Development data for a prospective version; real forecasts calculated from synthetic windows."""
from pathlib import Path
from kit.verify_diagnosis_traces import make_forecaster
from pinkprocess.kit_dataset import dataset_snapshot
from pinkprocess.kit_evidence import analyst_evidence
from lm_reference.flan_prompt import telemetry_prompt as original_prompt
from lm_reference import adapt_flan as training

forecast=make_forecaster(Path('profiles'))
original_example=training.example

def hint(e):
    obs=e.get('observations') or {}
    if not obs.get('peak_waiting') or not obs.get('first'): return ''
    first,peak,last=obs['first']['waiting'],obs['peak_waiting']['waiting'],e['latest']['waiting']
    if not peak: state='No backlog was observed in this window.'
    elif not last: state='The queue cleared by the final sample.'
    elif first<last: state=f'Active queue. Waiting rose from {first} to {last}.'
    elif first>last: state=f'Active queue. Waiting fell from {first} to {last}.'
    else: state=f'Active queue. First and final waiting counts were both {last}.'
    return state

def prompt(q,e):
    text=original_prompt(q,e);state=hint(e)
    return text.replace('\nQuestion:',f' Measured queue history: {state}\nQuestion:',1) if state else text

def example(rng,task,**kwargs):
    if rng.random()<0.18:
        return original_example(rng,task,**kwargs)
    n=rng.randint(50,190); k=rng.randint(1,40); height=rng.randint(2,32); floor=rng.randint(1,8)
    a=rng.randint(0,n//3);b=rng.randint(n//2,2*n//3)
    mode=rng.randrange(5);start=rng.choice((0,0,10000))
    rows=[]
    for i in range(n):
        if mode==0: w=0
        elif mode==1: w=max(0,height-abs(i-b))
        elif mode==2: w=max(0,(i-a)*height//(n-a))
        elif mode==3: w=floor
        else: w=max(floor,height-i*height//n)
        rows.append((start+i*100,k if w else rng.randrange(k+1),w))
    csv='time_ms,busy,waiting\n'+''.join('%d,%d,%d\n'%r for r in rows)
    snap=dataset_snapshot('dev','dev',csv,k,forecast);e=analyst_evidence(snap)
    e['origin']=rng.choice(('dataset','dataset','live','demo')); obs=e['observations'];peak=obs['peak_waiting'];last=rows[-1]
    q,a0=original_example(rng,task,**kwargs)
    q=q.split('\nQuestion: ')[-1].split('\nAnswer:')[0]
    if task==0:
        state=hint(e)
        pred=e['forecast_interpretation'].split('. ')[0].rstrip('.')
        answer=f'{state} The peak was {peak["waiting"]}. {pred}. Cause unknown. Measure arrivals and connection hold times.'
    elif task==1: answer=f'{last[2]} requests are waiting.'
    elif task==2: answer=f'The queue peaked at {peak["waiting"]} requests {(last[0]-peak["time_ms"])/1000:g} seconds ago.'
    elif task==3: answer=f'The mean queue was {obs["waiting"]["mean"]:.4g} requests.'
    elif task==4: answer=f'{last[1]} of {k} connections are busy.'
    elif task==5: answer='The measurements do not identify endpoints, latency or root cause.'
    elif task==6: answer=e['forecast_interpretation']
    elif task==7: answer=f'The pool capacity is {k} connections.'
    elif task==8: answer=f'Waiting went from {rows[0][2]} to {last[2]} requests.'
    elif task==9: answer=f'{obs["covered_ms"]/1000:g} seconds were recorded.'
    else: answer='The requested metric is not in the telemetry.'
    return prompt(q,e),answer

if __name__=='__main__':
 import sys
 directory=Path(sys.argv[1]); training.example=example; training.telemetry_prompt=prompt
 training.train(directory/'source',Path('build/flan/spiece.model'),directory/'candidate.safetensors',360,4227,True,True,True)
