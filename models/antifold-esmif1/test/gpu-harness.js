import {createGVPTransformerEngine,requestWebGPU} from '../webgpu/index.js';
import {compareOutput,checkReplay,oracleTensor} from './parity.js';
export async function run(){
 const log=async x=>fetch('/log',{method:'POST',body:JSON.stringify(x)}),cases=await(await fetch('/fixtures/cases.json')).json(),metrics={};
 const {device,adapter,info}=await requestWebGPU({useF16:true});
 const gpu={vendor:info.vendor,architecture:info.architecture,device:info.device,description:info.description,isFallbackAdapter:adapter.isFallbackAdapter,features:[...device.features]};
 if(!gpu.vendor&&!gpu.description)throw new Error('Hardware identity unavailable');await log(gpu);
 try{for(const variant of ['antifold','esmif1'])for(const precision of ['fp32','fp16','int8']){
  const loadStart=performance.now(),engine=await createGVPTransformerEngine({baseUrl:'/weights',variant,precision,device,useF16:false}),coldLoadMs=performance.now()-loadStart;
  try{for(const c of cases.filter(c=>c.variant===variant)){
   const meta=await(await fetch(`/fixtures/oracle_${c.name}_${precision}.json`)).json(),bytes=await(await fetch(`/fixtures/oracle_${c.name}_${precision}.bin`)).arrayBuffer(),input=c.sourceFile?await(await fetch('/'+c.sourceFile)).text():c.nativeSequence?{...c,sequence:c.nativeSequence}:c,result=await engine[c.nativeSequence?'score':'predict'](input,{returnIntermediates:true,...(c.nativeSequence?{sequence:c.sequence}:{})});
   const key=`${c.name}/${precision}`;metrics[key]={traces:compareOutput(result,c,meta,bytes),coldLoadMs,timingMs:result.timingMs};
   const sample=await engine.sample(input,{...meta.replay,uniforms:oracleTensor(meta,bytes,'replay.uniforms'),returnIntermediates:true});metrics[key].replay=checkReplay(sample,meta,bytes);
   if(meta.independent){const independent=await engine.sampleIndependent(c,{...meta.independent,uniforms:oracleTensor(meta,bytes,'independent.uniforms')});if(independent.tokens.toString()!==oracleTensor(meta,bytes,'independent.tokens').toString()||independent.selected.toString()!==oracleTensor(meta,bytes,'independent.mask').toString())throw new Error('Native-conditioned CDR replay mismatch');metrics[key].independentReplay=true;}
   if(engine.memoryUsage().transientBuffers!==0)throw new Error('Transient GPU buffer leak');
   if(c.benchmark){const warm=[];for(let i=0;i<3;i++)warm.push((await engine.predict(c)).timingMs);metrics[key].warmMs=warm;}
   await log({case:key,timingMs:result.timingMs,logits:metrics[key].traces.logits,replay:metrics[key].replay});
  }}finally{engine.destroy();engine.destroy();if(engine.memoryUsage().totalBytes!==0)throw new Error('Resident GPU buffer leak');}
  // An externally supplied device remains usable after engine.destroy().
  const probe=device.createBuffer({size:4,usage:GPUBufferUsage.COPY_DST});probe.destroy();
 }}finally{device.destroy();}
 return {gpu,metrics};
}
