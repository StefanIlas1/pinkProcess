import json
from collections import Counter
from pathlib import Path
from kit.profile_flan_tiles import CountedFlan, CountingFabric
from lm_reference.flan_prompt import telemetry_prompt
from pinkprocess.kit_evidence import analyst_evidence
from lm_reference.projection_overlay import QUERY_PROJECTIONS
report=json.loads(Path("kit/evidence/three-diagnosis-traces-physical.json").read_text())
case=report['cases'][2]
class FamilyFlan(CountedFlan):
    def linear(self, name,x):
        before=self.fabric.submissions
        result=super().linear(name,x)
        if name=='lm_head': family='output vocabulary'
        elif 'DenseReluDense' in name: family=f"{name.split('.')[0]} feed-forward"
        else: family=f"{name.split('.')[0]} attention"
        stats[family]+=self.fabric.submissions-before
        if name in QUERY_PROJECTIONS:
            stats_off['score submissions']+=self.fabric.submissions-before
            stats_off['projections']+=1
        return result
stats=Counter()
stats_off=Counter()
r=FamilyFlan(Path("build/flan/diagnostic"),CountingFabric());r.steps=[]
assert r.generate(telemetry_prompt(report['question'],analyst_evidence(case['started']['snapshot'])))==case['result']['answer']
assert sum(stats.values())==r.fabric.submissions
print(stats,stats_off)
