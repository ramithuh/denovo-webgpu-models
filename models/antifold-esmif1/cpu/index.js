import {loadWeights,decodeTensor,prepareInput,VOCAB,CANONICAL_TOKENS,regionMask} from '../shared/index.js';
import {encode,decode} from '../shared/model.js';
import {independentPlan,finishIndependent} from '../shared/sampling.js';
import {CPUOps} from './ops.js';
export function statistics(logits,tokens,mask){
 const L=tokens.length,V=35,logProbabilities=new Float32Array(L),canonicalProbabilities=new Float32Array(L*20),entropyPerplexity=new Float32Array(L);let total=0,count=0;
 for(let r=0;r<L;r++){
  const row=logits.subarray(r*V,(r+1)*V),max=Math.max(...row);let sum=0;for(const x of row)sum+=Math.exp(x-max);logProbabilities[r]=row[tokens[r]]-max-Math.log(sum);
  const c=Array.from(CANONICAL_TOKENS,t=>row[t]),m=Math.max(...c),s=c.reduce((a,v)=>a+Math.exp(v-m),0);let ent=0;
  for(let j=0;j<20;j++){const p=Math.exp(c[j]-m)/s;canonicalProbabilities[r*20+j]=p;ent-=p*Math.log(p);}entropyPerplexity[r]=Math.exp(ent);
  if(mask[r]){total+=logProbabilities[r];count++;}
 }
 return {logProbabilities,canonicalProbabilities,entropyPerplexity,logLikelihood:count?total/count:NaN,perplexity:count?Math.exp(-total/count):NaN};
}
export function inverseCDF(logits,u,temperature=1){if(!(u>=0&&u<1)||!(temperature>0&&Number.isFinite(temperature)))throw new Error('Invalid sampling draw or temperature');const max=Math.max(...logits);let sum=0;const p=Array.from(logits,v=>{const x=Math.exp((v-max)/temperature);sum+=x;return x;});let cumulative=0;for(let i=0;i<p.length;i++){cumulative+=p[i]/sum;if(u<cumulative)return i;}return p.length-1;}
export async function createGVPTransformerCPU({baseUrl,variant='antifold',precision='fp32',...options}={}){
 const bundle=await loadWeights(`${baseUrl.replace(/\/$/,'')}/${variant}/${precision}`,options),weights=Object.fromEntries(Object.entries(bundle.weights).map(([k,v])=>[k,{data:decodeTensor(v),shape:v.shape}]));
 const ops=new CPUOps(weights);let destroyed=false,busy=false,cached=null;
 const same=(a,b)=>a.length===b.length&&a.every((v,i)=>Object.is(v,b[i]));
 const encoder=async(p,trace,intermediates,signal)=>{if(!intermediates&&cached&&same(p.coords,cached.coords)&&same(p.confidence,cached.confidence))return cached.enc;const enc=await encode(ops,p,trace,intermediates,signal);cached={coords:p.coords.slice(),confidence:p.confidence.slice(),enc};return enc;};
 const enter=()=>{if(destroyed)throw new Error('Engine destroyed');if(busy)throw new Error('Inference already running');busy=true;};
 const engine={manifest:bundle.manifest,
  async predict(input,{returnIntermediates=false,signal,...opts}={}){
   enter();try{signal?.throwIfAborted();const start=performance.now(),p=prepareInput(input,opts),trace={},enc=await encoder(p,trace,returnIntermediates,signal),out=await decode(ops,enc,p.prev,trace,returnIntermediates,signal),logits=out.data;
    const mask=Int32Array.from(p.residues,(r,i)=>r&&p.tokens[i]!==30&&p.tokens[i]!==1&&(!p.targetChain||r.chain===p.targetChain)?1:0),stats=statistics(logits,p.tokens,mask),cm=Int32Array.from(mask,(x,i)=>x*p.coordMask[i+1]),coord=statistics(logits,p.tokens,cm);
    return {logits,...stats,coordinateLogLikelihood:coord.logLikelihood,coordinatePerplexity:coord.perplexity,sequence:p.sequence,residues:p.residues,timingMs:performance.now()-start,...(returnIntermediates?{intermediates:Object.fromEntries(Object.entries(trace).map(([k,v])=>[k,v.data]))}:{})};
   }finally{busy=false;}
  },
  async sample(input,{uniforms,temperature=1,fixed={},length,signal,returnIntermediates=false,...opts}={}){
   enter();try{const start=performance.now(),p=prepareInput(input,opts);length??=p.targetChain?p.residues.findIndex(r=>!r||r.chain!==p.targetChain):p.L;if(length<0)length=p.L;if(opts.regions){fixed={...fixed};const mask=regionMask(p.residues,{...p,...opts});for(let i=0;i<length;i++)if(!mask[i]&&fixed[i]===undefined)fixed[i]=p.tokens[i];}if(!Number.isInteger(length)||length<1||length>p.L||uniforms?.length!==length)throw new Error('Supply one uniform draw per sampled position');for(const u of uniforms)if(!(u>=0&&u<1))throw new Error('Invalid uniform draw');if(!(temperature>0&&Number.isFinite(temperature)))throw new Error('Invalid temperature');
    for(const [pos,t]of Object.entries(fixed))if(!Number.isInteger(+pos)||+pos<0||+pos>=length||(typeof t==='number'?t<0||t>=35||!Number.isInteger(t):!VOCAB.includes(t)))throw new Error('Invalid fixed position/token');
    const enc=await encoder(p,{},false,signal),tokens=new Int32Array(length+1);tokens[0]=33;const replayLogits=[];
    for(let i=0;i<length;i++){signal?.throwIfAborted();const row=(await decode(ops,enc,tokens.subarray(0,i+1),{},false,signal)).data.slice(i*35,(i+1)*35);replayLogits.push(...row);tokens[i+1]=fixed[i]===undefined?inverseCDF(row,uniforms[i],temperature):typeof fixed[i]==='number'?fixed[i]:VOCAB.indexOf(fixed[i]);if(tokens[i+1]<0||tokens[i+1]>=35)throw new Error('Invalid fixed token');}
    return {tokens:tokens.slice(1),sequence:Array.from(tokens.slice(1),t=>VOCAB[t]).join(''),timingMs:performance.now()-start,...(returnIntermediates?{logits:Float32Array.from(replayLogits)}:{})};
   }finally{busy=false;}
  },
  destroy(){destroyed=true;ops.weights={};cached=null;},memoryUsage(){return {bufferCount:0,totalBytes:0,transientBuffers:0,transientBytes:0};}
 };
 engine.score=engine.predict;
 engine.sampleIndependent=async(input,options={})=>{
  const plan=independentPlan(input,options),result=await engine.predict(input,options),draws=new Int32Array(plan.p.L*2);
  for(let i=0;i<plan.p.L;i++){const row=Array.from(CANONICAL_TOKENS,t=>result.logits[i*35+t]);draws[i*2]=inverseCDF(row,plan.uniforms[i],plan.temperature);draws[i*2+1]=row.indexOf(Math.max(...row));}
  return {...finishIndependent(plan,draws,options),timingMs:result.timingMs};
 };
 return engine;
}
export const createAntiFoldCPU=createGVPTransformerCPU;
export const createESMIF1CPU=options=>createGVPTransformerCPU({...options,variant:'esmif1'});
