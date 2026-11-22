import {mkdir,writeFile} from 'node:fs/promises';
import {runBrowser} from './browser.mjs';
for(const d of ['tmp','browser-cache','browser-config'])await mkdir(new URL('../.cache/'+d,import.meta.url),{recursive:true});
const result=await runBrowser('/test/gpu-long-suite.js',{timeoutMs:3600000});
if(!result.ok)throw Error(result.error);
await writeFile(new URL('../fixtures/gpu_long_metrics.json',import.meta.url),JSON.stringify(result,null,2)+'\n');
console.log('400-step final tensor and full trajectory parity passed',result.cases.map(c=>({name:c.name,timingMs:c.timingMs})));
