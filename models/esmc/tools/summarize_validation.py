"""Record final TAP results and runtime hashes from the actual local test runs."""
import datetime, hashlib, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 commands={}
 for command,filename in [('npm test','npm-test.log'),('npm run test:cpu','cpu-test.log'),('npm run test:gpu','gpu-test-final.log')]:
  log=(ROOT/'.cache'/filename).read_text()
  record={}
  for key in ['tests','pass','fail','cancelled','skipped','duration_ms']:
   matches=re.findall(r'^# '+key+r' ([0-9.]+)$',log,re.M)
   assert matches, f'No completed TAP summary: {command} ({key})'
   record[key]=float(matches[-1]) if key=='duration_ms' else int(matches[-1])
  assert record['tests']==record['pass'] and record['fail']==record['cancelled']==record['skipped']==0, (command,record)
  record['transcript_sha256']=hashlib.sha256(log.encode()).hexdigest();commands[command]=record
 runtime={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for directory in ['shared','cpu','webgpu'] for p in sorted((ROOT/directory).glob('*.js'))}
 result=dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),commands=commands,runtime_sha256=runtime)
 (ROOT/'fixtures/validation_summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(commands,indent=2))
if __name__=='__main__':main()
