import {loadWeights,prepareInput,regionMask,VOCAB,CANONICAL_TOKENS} from '../shared/index.js';
import {encode,decode} from '../shared/model.js';
import {independentPlan,finishIndependent} from '../shared/sampling.js';
import {GPUOps} from './ops.js';
export async function requestWebGPU({useF16=false,allowSoftwareFallback=true}={}){
 if(!globalThis.navigator?.gpu)throw new Error('Hardware WebGPU is required (HTTPS or localhost)');
 const adapter=(await navigator.gpu.requestAdapter({powerPreference:'high-performance'}))??(await navigator.gpu.requestAdapter());if(!adapter)throw new Error('No WebGPU adapter');
 const raw=adapter.info??await adapter.requestAdapterInfo?.();const info={vendor:raw?.vendor,architecture:raw?.architecture,device:raw?.device,description:raw?.description,isFallbackAdapter:raw?.isFallbackAdapter??adapter.isFallbackAdapter};
 const isSoftware=info.isFallbackAdapter||/swiftshader|llvmpipe|lavapipe|software|microsoft basic|\bwarp\b/i.test(JSON.stringify(info));
 if(isSoftware&&!allowSoftwareFallback)throw new Error('Software WebGPU adapters are rejected');
 const device=await adapter.requestDevice({requiredFeatures:useF16&&adapter.features.has('shader-f16')?['shader-f16']:[],requiredLimits:{maxBufferSize:adapter.limits.maxBufferSize,maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize}});
 return {device,adapter,info};
}
async function outputStats(ops,logits,p){
 const tokens=ops.tensor(p.tokens,[p.L,1]);
 const packed=await ops.elem([p.L,22],`let r=i/22u;let j=i%22u;var m=-3.402823e38;for(var t=0u;t<35u;t++){m=max(m,x0[r*35u+t]);}var s=0.0;for(var t=0u;t<35u;t++){s+=exp(x0[r*35u+t]-m);}if(j==0u){y[i]=x0[r*35u+u32(x1[r])]-m-log(s);return;}
 var aa:array<u32,20>=array<u32,20>(${Array.from(CANONICAL_TOKENS,t=>t+'u').join(',')});var cm=-3.402823e38;for(var t=0u;t<20u;t++){cm=max(cm,x0[r*35u+aa[t]]);}var cs=0.0;for(var t=0u;t<20u;t++){cs+=exp(x0[r*35u+aa[t]]-cm);}if(j>=2u){y[i]=exp(x0[r*35u+aa[j-2u]]-cm)/cs;return;}var entropy=0.0;for(var t=0u;t<20u;t++){let prob=exp(x0[r*35u+aa[t]]-cm)/cs;entropy-=prob*log(max(prob,1e-30));}y[i]=exp(entropy);`,[logits,tokens]);
 const mask=ops.tensor(Int32Array.from(p.residues,(r,i)=>r&&p.tokens[i]!==30&&p.tokens[i]!==1&&(!p.targetChain||r.chain===p.targetChain)?1:0),[p.L,1]),cm=ops.tensor(p.coordMask.slice(1,-1),[p.L,1]);
 const scores=await ops.elem([1,8],`var sum=0.0;var count=0.0;for(var r=0u;r<${p.L}u;r++){if(x1[r]!=0&&(i%4u<2u||x2[r]!=0)){sum+=x0[r*22u];count+=1.0;}}let mean=sum/max(count,1.0);y[i]=select(select(mean,exp(-mean),i%2u==1u),count,i>=4u);`,[packed,mask,cm]);
 return {packed,scores};
}
export async function createGVPTransformerEngine({baseUrl,variant='antifold',precision='fp32',device,useF16=false,signal,onProgress,cache='none',read}={}){
 if(!baseUrl)throw new Error('baseUrl must contain variant/precision bundles');if(!['antifold','esmif1'].includes(variant)||!['fp32','fp16','int8'].includes(precision))throw new Error('Unknown variant or precision');signal?.throwIfAborted();
 const ownsDevice=!device;let info;if(!device)({device,info}=await requestWebGPU({useF16}));
 const ops=new GPUOps(device,{ownsDevice,useF16,onProgress});let busy=false;
 try{
 const {manifest}=await loadWeights(`${baseUrl.replace(/\/$/,'')}/${variant}/${precision}`,{read,cache,signal,onProgress,onTensor:(name,t)=>ops.upload(name,t)});
 if(manifest.model!==variant||manifest.precision!==precision)throw new Error('Requested model/precision does not match bundle');
 const enter=()=>{if(busy)throw new Error('Inference already running');ops.begin();busy=true;};
 const engine={manifest,adapterInfo:info,
 async predict(input,{returnIntermediates=false,signal,...options}={}){
  signal?.throwIfAborted();const p=prepareInput(input,options);ops.admit(p.N,p.L);enter();const started=performance.now();
  try{
   const trace={},enc=await encode(ops,p,trace,returnIntermediates,signal),logits=await decode(ops,enc,p.prev,trace,returnIntermediates,signal),stats=await outputStats(ops,logits,p);
   const values=await ops.readMany({...(returnIntermediates?trace:{}),logits,packed:stats.packed,scores:stats.scores});signal?.throwIfAborted();if(values.logits.some(v=>!Number.isFinite(v)))throw new Error('Nonfinite GPU logits');
   const logProbabilities=new Float32Array(p.L),entropyPerplexity=new Float32Array(p.L),canonicalProbabilities=new Float32Array(p.L*20);for(let r=0;r<p.L;r++){logProbabilities[r]=values.packed[r*22];entropyPerplexity[r]=values.packed[r*22+1];canonicalProbabilities.set(values.packed.subarray(r*22+2,(r+1)*22),r*20);}
   const scores=values.scores;for(let i=0;i<4;i++)if(scores[i+4]===0)scores[i]=NaN;delete values.packed;delete values.scores;
   return {logits:values.logits,logProbabilities,entropyPerplexity,canonicalProbabilities,logLikelihood:scores[0],perplexity:scores[1],coordinateLogLikelihood:scores[2],coordinatePerplexity:scores[3],sequence:p.sequence,residues:p.residues,timingMs:performance.now()-started,...(returnIntermediates?{intermediates:values}:{})};
  }finally{try{await ops.finish();}finally{busy=false;}}
 },
 async sample(input,{uniforms,temperature=1,fixed={},length,signal,returnIntermediates=false,...options}={}){
  signal?.throwIfAborted();const p=prepareInput(input,options);length??=p.targetChain?p.residues.findIndex(r=>!r||r.chain!==p.targetChain):p.L;if(length<0)length=p.L;if(options.regions){fixed={...fixed};const mask=regionMask(p.residues,{...p,...options});for(let i=0;i<length;i++)if(!mask[i]&&fixed[i]===undefined)fixed[i]=p.tokens[i];}if(!Number.isInteger(length)||length<1||length>p.L||uniforms?.length!==length||Array.from(uniforms).some(u=>!(u>=0&&u<1)))throw new Error('Supply one uniform draw in [0,1) per position');if(!(temperature>0&&Number.isFinite(temperature)))throw new Error('Invalid temperature');
  for(const [pos,t]of Object.entries(fixed))if(!Number.isInteger(+pos)||+pos<0||+pos>=length||(typeof t==='number'?t<0||t>=35||!Number.isInteger(t):!VOCAB.includes(t)))throw new Error('Invalid fixed position/token');
  ops.admit(p.N,length);enter();const started=performance.now();
  try{
   const enc=await encode(ops,p,{},false,signal),tokens=new Int32Array(length+1);tokens[0]=33;const replayLogits=[];
   for(let i=0;i<length;i++){
    signal?.throwIfAborted();const logits=await decode(ops,enc,tokens.subarray(0,i+1),{},false,signal);
    const sampled=await ops.elem([1,1],`var aa:array<u32,20>=array<u32,20>(${Array.from(CANONICAL_TOKENS,t=>t+'u').join(',')});var m=-3.402823e38;for(var j=0u;j<20u;j++){m=max(m,x0[${i*35}u+aa[j]]);}var sum=0.0;for(var j=0u;j<20u;j++){sum+=exp((x0[${i*35}u+aa[j]]-m)/${temperature});}var cumulative=0.0;var token=aa[19u];for(var j=0u;j<20u;j++){cumulative+=exp((x0[${i*35}u+aa[j]]-m)/${temperature})/sum;if(${uniforms[i]}<cumulative){token=aa[j];break;}}y[0]=f32(token);`,[logits]);
    const values=await ops.readMany({sampled,...(returnIntermediates?{logits}:{})});signal?.throwIfAborted();tokens[i+1]=fixed[i]===undefined?values.sampled[0]:typeof fixed[i]==='number'?fixed[i]:VOCAB.indexOf(fixed[i]);if(returnIntermediates)replayLogits.push(...values.logits.subarray(i*35,(i+1)*35));ops.keep([enc.x,enc.padding]);
   }
   return {tokens:tokens.slice(1),sequence:Array.from(tokens.slice(1),t=>VOCAB[t]).join(''),timingMs:performance.now()-started,...(returnIntermediates?{logits:Float32Array.from(replayLogits)}:{})};
  }finally{try{await ops.finish();}finally{busy=false;}}
 },
 memoryUsage:()=>ops.memoryUsage(),destroy:()=>ops.destroy()};
 engine.score=engine.predict;
 engine.sampleIndependent=async(input,options={})=>{
  const plan=independentPlan(input,options),result=await engine.predict(input,options);enter();
  try{const x=ops.tensor(result.logits,[plan.p.L,35]),u=ops.tensor(Float32Array.from(plan.uniforms),[plan.p.L,1]);
   const draws=await ops.elem([plan.p.L,2],`let r=i/2u;var aa:array<u32,20>=array<u32,20>(${Array.from(CANONICAL_TOKENS,t=>t+'u').join(',')});var m=-3.402823e38;var arg=0u;for(var j=0u;j<20u;j++){let v=x0[r*35u+aa[j]];if(v>m){m=v;arg=j;}}if(i%2u==1u){y[i]=f32(arg);return;}var sum=0.0;for(var j=0u;j<20u;j++){sum+=exp((x0[r*35u+aa[j]]-m)/${plan.temperature});}var c=0.0;var token=19u;for(var j=0u;j<20u;j++){c+=exp((x0[r*35u+aa[j]]-m)/${plan.temperature})/sum;if(x1[r]<c){token=j;break;}}y[i]=f32(token);`,[x,u]);
   const values=await ops.read(draws);options.signal?.throwIfAborted();return {...finishIndependent(plan,values,options),timingMs:result.timingMs};
  }finally{try{await ops.finish();}finally{busy=false;}}
 };
 return engine;
 }catch(e){ops.destroy();throw e;}
}
export const createAntiFoldEngine=createGVPTransformerEngine;
export const createESMIF1Engine=options=>createGVPTransformerEngine({...options,variant:'esmif1'});
