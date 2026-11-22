// This diagnostic deliberately fails when the requested non-gating stress matrix exceeds tolerance.
import {browserTask} from '../test/browser.mjs';
import {root} from '../test/helpers.mjs';
import {writeFile} from 'node:fs/promises';
const result=await browserTask(page=>page.evaluate(async()=>{
 const {createFrameFlowEngine}=await import('/webgpu/index.js'),{fixture}=await import('/test/parity.js'),{diffStats}=await import('/shared/weights.js');const read=async u=>{const r=await fetch(u);if(!r.ok)throw Error('Generate stress fixtures first: '+u);return r.arrayBuffer();};const matrix=JSON.parse(new TextDecoder().decode(await read('/fixtures/stress_matrix.json'))),rows=[];
 for(const precision of ['fp32','fp16','int8']){const e=await createFrameFlowEngine({baseUrl:'/weights',precision,allowSoftwareFallback:false});try{for(const name of matrix.cases.filter(n=>n.startsWith('stress_'+precision+'_'))){const f=await fixture(name,read,'/');const out=f.meta.kind==='forward'?await e.forward(f.input):await e.sample({initial:f.input,steps:f.meta.steps});const errors={};for(const k of ['pred_trans','pred_rotmats','coords'])errors[k]=diffStats(out[k],f.output[k]);rows.push({name,errors});}}finally{e.destroy();}}return rows;
}));await writeFile(root+'fixtures/stress_metrics.json',JSON.stringify(result,null,2)+'\n');const failures=result.filter(r=>Object.values(r.errors).some(v=>v.relRms>=1e-4));console.log(JSON.stringify(failures,null,2));if(failures.length)process.exitCode=1;
