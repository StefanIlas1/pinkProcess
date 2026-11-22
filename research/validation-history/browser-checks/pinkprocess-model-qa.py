import json
from pathlib import Path
import sys
from time import monotonic
from uuid import uuid4
from urllib.request import urlopen
from pinkprocess import live_analyst
from pinkprocess.kit_app import DIAGNOSIS_EXAMPLE, analyst_evidence
from pinkprocess.telemetry_stream import TelemetryStream

live_analyst.MODEL = sys.argv[1]
recovered_path = Path('build/live-kit/analyst-input.json')
recovered = json.loads(recovered_path.read_text())

def make_pressure_fixture():
    def scorer(window, capacity):
        return {'model':'queue_outlook', 'version':'queue-outlook-v2', 'horizon_ms': 10000,
                'scores': [30, 15], 'time_ms': window[-1].time_ms}
    stream = TelemetryStream(scorer, clock=lambda: 0)
    stream.ingest({'session':str(uuid4()), 'source':'Fixture with rising queue pressure', 'origin':'demo',
                   'capacity':12, 'readings':[{'time_ms': i*100, 'busy': 12, 'waiting': i//10} for i in range(100)]})
    return stream.query({'op':'stream_evidence','window_ms':10000})

pressure = make_pressure_fixture()
cases = [
    ('recovered', 'captured board SQLite demo telemetry', recovered, DIAGNOSIS_EXAMPLE,
     ['2 of 4 connections occupied and 0 requests waiting in the latest reading',
      'Peak of 117 requests waiting about 50.8 seconds earlier',
      'FPGA warns that requests may start waiting within the next 10 seconds',
      'Identifies telemetry as a demo and gives a practical next check without diagnosing a cause']),
    ('increasing', 'controlled telemetry fixture with simulated FPGA forecast', pressure, DIAGNOSIS_EXAMPLE,
     ['12 of 12 connections occupied and 9 requests waiting in the latest reading',
      'Queue grew from 0 to 9 waiting over the observation window',
      'FPGA estimate keeps queue nonempty, with more than 9 waiting at end of 10-second horizon',
      'Identifies telemetry as a demo and gives a practical next check without diagnosing a cause']),
    ('cause_unavailable', 'captured board SQLite demo telemetry', recovered,
     'Which endpoint caused this problem? What was its P95 wait time?',
     ['Endpoint attribution and P95 wait time are unavailable from pool counters']),
    ('average', 'controlled telemetry fixture with simulated FPGA forecast', pressure,
     'What is the mean waiting queue over the most recent ten seconds?',
     ['Mean of 4.5 waiting requests']),
]
results = []
for name, source, snapshot, question, expected in cases:
    evidence = analyst_evidence(snapshot)
    start = monotonic()
    result = live_analyst.analyze(question, evidence)
    elapsed = round(monotonic() - start, 3)
    results.append(dict(name=name, source=source, question=question, evidence=evidence,
                        expected_facts=expected, elapsed_seconds=elapsed, result=result))
    print(name, result['answer'] or result['errors'], f'{elapsed}s', flush=True)
if len(sys.argv) > 2:
    with urlopen('http://127.0.0.1:11434/api/tags', timeout=5) as response:
        model = next(item for item in json.load(response)['models'] if item['model'] == live_analyst.MODEL)
    with urlopen('http://127.0.0.1:11434/api/version', timeout=5) as response:
        version = json.load(response)['version']
    Path(sys.argv[2]).write_text(json.dumps({'model': model, 'ollama_version': version, 'cases': results}, indent=2) + '\n')
