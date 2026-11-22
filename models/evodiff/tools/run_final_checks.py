"""Run the three requested commands separately, retain logs and exact exit status."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
env = dict(os.environ)
env['PATH'] = '/home/ruh/.local/bin:/home/ruh/.nvm/versions/node/v22.21.1/bin:' + env['PATH']
env['npm_config_cache'] = str(ROOT / '.cache/npm')
results = []
for command, name in [(['npm', 'test'], 'npm-test'), (['npm', 'run', 'test:cpu'], 'cpu'), (['npm', 'run', 'test:gpu'], 'gpu')]:
    start = time.monotonic()
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    log = ROOT / '.cache' / f'final-{name}.log'
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    row = {'command': ' '.join(command), 'exitCode': result.returncode, 'seconds': time.monotonic()-start, 'startedUTC': timestamp, 'log': str(log.relative_to(ROOT))}
    results.append(row)
    (ROOT / 'fixtures/test_runs.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(row), flush=True)
    if result.returncode:
        print(log.read_text()[-12000:], flush=True)
        raise SystemExit(result.returncode)
