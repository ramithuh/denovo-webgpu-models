import {loadWeights,decodeTensor,tokenize} from '../../shared/dyna1/index.js';
import {matmulSource,normSource,elementSource,gatherSource,ropeSource,scoresSource,softmaxSource,contextSource} from './shaders.js';

export async function requestWebGpuDevice({useF16=true}={}) {
  if(!globalThis.navigator?.gpu) throw new Error('WebGPU requires a supported browser and HTTPS or localhost');
  const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
  if(!adapter) throw new Error('No WebGPU adapter');
  const hasF16=adapter.features.has('shader-f16');
  const device=await adapter.requestDevice({requiredFeatures:useF16&&hasF16?['shader-f16']:[]});
  return {adapter,device,hasF16};
}
class Engine {
 constructor(device,{ownsDevice=false,useF16=true}={}) {
  this.device=device;this.ownsDevice=ownsDevice;this.useF16=useF16&&device.features.has('shader-f16');this.weights={};this.buffers=new Set();this.pipelines=new Map();this.destroyed=false;this.busy=false;this.lost=null;
  device.lost.then(info=>{this.lost=info.message||'Device lost';});
  this.zero=this.buffer(new Float32Array(4));
 }
 buffer(dataOrSize,temporary=false) {
  const data=typeof dataOrSize==='number'?null:dataOrSize;
  const size=Math.max(4,Math.ceil((data?.byteLength??dataOrSize)/4)*4);
  if(this.destroyed) throw new Error('Engine destroyed');
  const buffer=(temporary&&!data?this.pool.get(size)?.pop():null)??this.device.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  if(data) {const bytes=new Uint8Array(size);bytes.set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));this.device.queue.writeBuffer(buffer,0,bytes);}
  this.buffers.add(buffer);if(temporary) {this.temporary.add(buffer);this.liveTemps.add(buffer);}return buffer;
 }
 releaseExcept(current,traces) {
  const keep=new Set([current,...Object.values(traces).map(t=>t.buffer)]);
  for(const b of this.liveTemps) if(!keep.has(b)){if(!this.pool.has(b.size))this.pool.set(b.size,[]);this.pool.get(b.size).push(b);this.liveTemps.delete(b);}
 }
 memoryUsage(){return {bufferCount:this.buffers.size,totalBytes:[...this.buffers].reduce((n,b)=>n+b.size,0)};}
 async upload(name,t) {
  if(this.destroyed) throw new Error('Engine destroyed');
  if(name==='embedding.weight') t={dtype:'F32',shape:t.shape,bytes:new Uint8Array(decodeTensor(t).buffer)};
  this.weights[name]={buffer:this.buffer(t.bytes),dtype:t.dtype,shape:t.shape,scales:t.scales?this.buffer(t.scales):this.zero};
 }
 async pipeline(code,count) {
  let promise=this.pipelines.get(code);
  if(!promise) {
   promise=(async()=>{
    const module=this.device.createShaderModule({code});const info=await module.getCompilationInfo();
    const errors=info.messages.filter(m=>m.type==='error');if(errors.length) throw new Error(errors.map(e=>e.message).join('\n'));
    const layout=this.device.createBindGroupLayout({entries:Array.from({length:count},(_,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===count-1?'storage':'read-only-storage'}}))});
    const pipeline=await this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:'main'}});
    return {pipeline,layout};
   })();this.pipelines.set(code,promise);
  }
  return promise;
 }
 async dispatch(code,inputs,n,groups) {
  const y=this.buffer(n*4,true),buffers=[...inputs,y];const {pipeline,layout}=await this.pipeline(code,buffers.length);
  const bind=this.device.createBindGroup({layout,entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))});
  const pass=this.encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(...groups);pass.end();return y;
 }
 async predict(sequence,{returnIntermediates=false,signal}={}) {
  if(this.destroyed) throw new Error('Engine destroyed');if(this.lost) throw new Error(this.lost);if(this.busy) throw new Error('A prediction is already running');
  signal?.throwIfAborted();
  const {tokens,residues}=tokenize(sequence,this.config.max_sequence_length),L=tokens.length,D=this.config.d_model;
  this.busy=true;this.temporary=new Set();this.liveTemps=new Set();this.pool=new Map();this.encoder=this.device.createCommandEncoder();
  const started=performance.now(),traces={},w=this.weights;
  this.device.pushErrorScope('validation');
  const save=(name,value)=>{if(returnIntermediates) traces[name]={buffer:value,n:L*D};};
  const elem=(n,exp,inputs)=>this.dispatch(elementSource(n,exp,inputs.length),inputs,n,[Math.ceil(n/256)]);
  const ln=(x,p,{stride=D,offset=0,eps=1e-5}={})=>this.dispatch(normSource(L,D,{stride,offset,eps,bias:!!w[p+'.bias']}),[x,w[p+'.weight'].buffer,w[p+'.bias']?.buffer??this.zero],L*D,[L]);
  const linear=(x,p,K=D,N=D,raw=false)=>{
   const weight=w[p+(raw?'_weight':'.weight')],bias=w[p+(raw?'_bias':'.bias')];
   return this.dispatch(matmulSource(L,N,K,weight.dtype,!!bias,this.useF16),[x,weight.buffer,bias?.buffer??this.zero,weight.scales],L*N,[Math.ceil(N/16),Math.ceil(L/16)]);
  };
  const residual=(x,y,scale=1)=>elem(L*D,`y[i]=x[i]+b[i]/${scale};`,[x,y]);
  const attn=async(q,k,qkv,H,layout)=>{
   const scores=await this.dispatch(scoresSource(L,D,H,layout),[q,k],H*L*L,[Math.min(65535,Math.ceil(H*L*L/64)),Math.ceil(H*L*L/64/65535)]);
   const prob=await this.dispatch(softmaxSource(L,H),[scores],H*L*L,[H*L]);
   return this.dispatch(contextSource(L,D,H,{stride:3*D,offset:2*D}),[prob,qkv],L*D,[Math.ceil(L*D/64)]);
  };
  try {
   let x=await this.dispatch(gatherSource(L,D),[this.buffer(tokens,true),w['embedding.weight'].buffer],L*D,[Math.ceil(L*D/256)]);save('embedding',x);
   for(let i=0;i<this.config.backbone_layers;i++) {
    signal?.throwIfAborted();if(this.destroyed) throw new Error('Engine destroyed');
    const p=`backbone.${i}`,F=this.config.ffn_size,H=this.config.backbone_heads;
    const qkv=await linear(await ln(x,p+'.attn.layernorm_qkv.0'),p+'.attn.layernorm_qkv.1',D,3*D);
    let q=await ln(qkv,p+'.attn.q_ln',{stride:3*D}),k=await ln(qkv,p+'.attn.k_ln',{stride:3*D,offset:D});
    q=await this.dispatch(ropeSource(L,D,H),[q],L*D,[Math.ceil(L*D/256)]);k=await this.dispatch(ropeSource(L,D,H),[k],L*D,[Math.ceil(L*D/256)]);
    x=await residual(x,await linear(await attn(q,k,qkv,H),p+'.attn.out_proj'),this.config.residual_scale);
    const ff=await linear(await ln(x,p+'.ffn.0'),p+'.ffn.1',D,2*F);
    const act=await elem(L*F,`let r=i/${F}u;let j=i%${F}u;let a=x[r*${2*F}u+j];y[i]=(a/(1.0+exp(-a)))*x[r*${2*F}u+${F}u+j];`,[ff]);
    x=await residual(x,await linear(act,p+'.ffn.3',F,D),this.config.residual_scale);save(`backbone.${i}`,x);this.releaseExcept(x,traces);
   }
   save('final_embeddings',x);x=await ln(x,'head.norm1',{eps:1e-12});
   for(let i=0;i<this.config.head_layers;i++) {
    signal?.throwIfAborted();const p=`head.encoder.layers.${i}`,F=this.config.head_ffn_size;
    const qkv=await linear(await ln(x,p+'.norm1'),p+'.self_attn.in_proj',D,3*D,true);
    // Classifier attention consumes the projection directly. Its Q and K need no
    // separate normalization or rotary transformation before the dot product.
    const context=await attn(qkv,qkv,qkv,this.config.head_heads,{packedQkv:true});
    x=await residual(x,await linear(context,p+'.self_attn.out_proj'));
    const a=await linear(await ln(x,p+'.norm2'),p+'.linear1',D,F);
    const relu=await elem(L*F,'y[i]=max(x[i],0.0);',[a]);
    x=await residual(x,await linear(relu,p+'.linear2',F,D));save(`head.${i}`,x);this.releaseExcept(x,traces);
   }
   const logits=await linear(await ln(x,'head.norm2',{eps:1e-12}),'head.output_layer',D,1);
   const probs=await elem(L,`y[i]=1.0/(1.0+exp(-(x[i]-(${Math.log(.05/.95)}))));`,[logits]);
   traces.logits={buffer:logits,n:L};traces.probs={buffer:probs,n:L};
   const size=Object.values(traces).reduce((a,t)=>a+t.n*4,0);
   const readback=this.device.createBuffer({size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});this.buffers.add(readback);this.temporary.add(readback);
   let offset=0;for(const t of Object.values(traces)){t.offset=offset;this.encoder.copyBufferToBuffer(t.buffer,0,readback,offset,t.n*4);offset+=t.n*4;}
   this.device.queue.submit([this.encoder.finish()]);
   await readback.mapAsync(GPUMapMode.READ);const mapped=readback.getMappedRange(),values={};
   for(const [name,t] of Object.entries(traces)) values[name]=new Float32Array(mapped.slice(t.offset,t.offset+t.n*4));
   readback.unmap();signal?.throwIfAborted();
   for(const array of [values.probs,values.logits]) for(const v of array) if(!Number.isFinite(v)) throw new Error('Nonfinite GPU output');
   return {probs:values.probs,logits:values.logits,residues,timingMs:performance.now()-started,...(returnIntermediates?{intermediates:{tokens,...values}}:{})};
  } finally {
   const validation=await this.device.popErrorScope();
   for(const b of this.temporary){b.destroy();this.buffers.delete(b);}this.temporary.clear();this.liveTemps.clear();this.pool.clear();this.busy=false;
   if(validation) throw new Error(`WebGPU validation: ${validation.message}`);
  }
 }
 destroy(){if(this.destroyed)return;this.destroyed=true;for(const b of this.buffers)b.destroy();this.buffers.clear();this.weights={};this.pipelines.clear();if(this.ownsDevice)this.device.destroy();}
}
export async function createDyna1Engine({baseUrl,precision='int8',device,onProgress,signal,useF16=true,cache='cache',read}={}) {
 if(!baseUrl) throw new Error('baseUrl must point to the directory containing precision bundles');
 if(!['fp32','fp16','int8'].includes(precision)) throw new Error('Unknown precision');
 const ownsDevice=!device;if(!device) ({device}=await requestWebGpuDevice({useF16}));
 const engine=new Engine(device,{ownsDevice,useF16});
 try {
  const bundle=await loadWeights(`${baseUrl.replace(/\/$/,'')}/${precision}`,{onProgress,signal,cache,read,onTensor:(name,t)=>engine.upload(name,t)});
  engine.config=bundle.config;engine.manifest=bundle.manifest;return engine;
 } catch(e){engine.destroy();throw e;}
}
