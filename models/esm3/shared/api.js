import {prepareInput,decodeSequence} from './input.js';
import {forward,decodeStructure,encodeStructure,OUTPUT_DIMS} from './model.js';
export class ESM3Engine {
 constructor(backend,{gpu=false,precision='fp32'}={}){this.b=backend;this.gpu=gpu;this.precision=precision;}
 async _run(input,kind,options={}){const {b}=this,start=performance.now();b.check();options.signal?.throwIfAborted();
  const run=async(save)=>{
   const p=kind==='decode'?input:prepareInput(input),L=p.L;
   if(this.gpu){if(kind==='decode')p.tokens=b.buffer(Float32Array.from(p.tokens),true);else for(const [k,v]of Object.entries(p))if(ArrayBuffer.isView(v))p[k]=b.buffer(Float32Array.from(v),true);}
   let result;
   if(kind==='forward')result=await forward(b,p,{...options,save});
   else if(kind==='encode'){const r=await encodeStructure(b,p,{...options,save});result={structure_tokens:r.tokens,frames:r.frames};}
   else result=await decodeStructure(b,p.tokens,L,{...options,save});
   if(!this.gpu)return result;
   const dims=kind==='forward'?OUTPUT_DIMS:kind==='encode'?{structure_tokens:1,frames:13}:{coordinates:9,affine:12,plddt:1,pairwise_logits:L*224,predicted_aligned_error:L,ptm:1/L};
   return Object.fromEntries(Object.entries(result).map(([name,buffer])=>[name,{buffer,n:L*dims[name]}]));
  };
  let result;
  if(this.gpu)result=await b.run(()=>run((...a)=>b.save(...a)),options);
  else {if(b.busy)throw new Error('Inference already running');b.busy=true;const traces={};try{result=await run((k,v)=>{if(options.returnIntermediates)traces[k]=v;});if(options.returnIntermediates)result.intermediates=traces;}finally{b.busy=false;}}
  return {...result,timingMs:performance.now()-start};
 }
 forward(input,options){return this._run(input,'forward',options);}
 encodeStructure(input,options){return this._run(input,'encode',options);}
 decodeStructure(tokens,options){if(!tokens||tokens.length<3||tokens[0]!==4098||tokens[tokens.length-1]!==4097||Array.from(tokens).some(t=>!Number.isInteger(t)||t<0||t>4100))throw new Error('Decoder tokens require BOS=4098, EOS=4097 and valid structure IDs');return this._run({tokens:Float32Array.from(tokens),L:tokens.length},'decode',options);}
 async generate(input,{track='sequence',steps=8,temperature=0,seed=0,signal,onStep,...options}={}){
  if(!['sequence','structure'].includes(track))throw new Error('Generation track must be sequence or structure');
  if(!Number.isInteger(steps)||steps<1||!Number.isFinite(temperature)||temperature<0)throw new Error('Invalid sampling configuration');
  const p=prepareInput(input),key=track+'_tokens',mask=track==='sequence'?32:4096;
  if(p.encodeStructure){const e=await this.encodeStructure(p,{signal});p.structure_tokens=Int32Array.from(e.structure_tokens);p.encodeStructure=false;}
  const pending=Array.from(p[key],(v,i)=>v===mask?i:-1).filter(i=>i>=0),initial=pending.length,history=[];
  let rng=seed>>>0;const random=()=>{rng=(rng+0x6D2B79F5)>>>0;let t=rng;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};
  for(let step=0;step<steps&&pending.length;step++){
   signal?.throwIfAborted();const result=await this.forward(p,{signal,...options}),logits=result[track+'_logits'],N=track==='sequence'?64:4096,min=track==='sequence'?4:0,max=track==='sequence'?24:4096;
   const candidates=pending.map(i=>{let best=-Infinity,id=min;for(let t=min;t<max;t++)if(logits[i*N+t]>best){best=logits[i*N+t];id=t;}
    let den=0;const probs=[];for(let t=min;t<max;t++){const v=Math.exp((logits[i*N+t]-best)/(temperature||1));probs.push(v);den+=v;}
    if(temperature>0){let u=random()*den;for(let t=min;t<max;t++){u-=probs[t-min];if(u<=0){id=t;break;}}}
    return {i,id,confidence:probs[id-min]/den};}).sort((a,b)=>b.confidence-a.confidence||a.i-b.i);
   const remaining=Math.floor(initial*Math.cos((step+1)/steps*Math.PI/2));const count=Math.max(1,pending.length-remaining);const chosen=candidates.slice(0,count);
   for(const {i,id}of chosen){p[key][i]=id;pending.splice(pending.indexOf(i),1);}const record={step,tokens:Array.from(p[key]),filled:chosen.map(x=>x.i)};history.push(record);await onStep?.(record);
  }
  const output={sequence:decodeSequence(p.sequence_tokens),sequence_tokens:p.sequence_tokens,structure_tokens:p.structure_tokens,history};
  if(track==='structure')Object.assign(output,await this.decodeStructure(p.structure_tokens,{signal,...options}));return output;
 }
 memoryUsage(){return this.b.memoryUsage();}
 destroy(){this.b.destroy();}
}
