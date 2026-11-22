import {readFile,writeFile} from 'node:fs/promises';
const commands=[['npm test','test-all-final.log'],['npm run test:cpu','test-cpu-final.log'],['npm run test:gpu','test-gpu-final-complete.log']];
const results=[];
for(const [command,file] of commands){
  const text=await readFile(new URL('../.cache/'+file,import.meta.url),'utf8'),result={command};
  for(const key of ['tests','pass','fail','cancelled','skipped','todo','duration_ms']){
    const match=text.match(new RegExp('^# '+key+' ([0-9.]+)$','m'));if(!match)throw Error(`Incomplete log ${file}: missing ${key}`);result[key]=Number(match[1]);
  }
  if(result.fail||result.cancelled||result.skipped||result.pass!==result.tests)throw Error(`Acceptance failed: ${command}`);
  results.push(result);
}
await writeFile(new URL('../fixtures/test_results.json',import.meta.url),JSON.stringify({recorded_at:new Date().toISOString(),results},null,2)+'\n');
console.log(results);
