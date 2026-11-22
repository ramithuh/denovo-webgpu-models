import * as S from './shaders.js';
import {decodeTensor} from '../shared/weights.js';
export async function requestWebGpuDevice({useF16=false}={}) {
 if(!globalThis.navigator?.gpu)throw new Error('Hardware WebGPU requires HTTPS or localhost');
 const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw new Error('No WebGPU adapter');
 const info=adapter.info??{};
 if(adapter.isFallbackAdapter||info.isFallbackAdapter||/swiftshader|llvmpipe|software|lavapipe/i.test(JSON.stringify({vendor:info.vendor,device:info.device,description:info.description})))throw new Error('Software WebGPU adapters are unsupported');
 const device=await adapter.requestDevice({requiredFeatures:useF16&&adapter.features.has('shader-f16')?['shader-f16']:[]});
 return {device,adapter};
}
export class GPUBackend {
 constructor(device,{ownsDevice=false,useF16=false}={}){const info=device.adapterInfo??{};if(info.isFallbackAdapter||/swiftshader|llvmpipe|software|lavapipe/i.test([info.vendor,info.device,info.description].join(' ')))throw new Error('Software WebGPU adapters are unsupported');this.device=device;this.ownsDevice=ownsDevice;this.useF16=useF16&&device.features.has('shader-f16');this.w={};this.buffers=new Set();this.pipelines=new Map();this.destroyed=false;this.busy=false;this.lost=null;device.lost.then(x=>{this.lost=x.message;});this.zero=this.buffer(new Float32Array(4));}
 check(){if(this.destroyed)throw new Error('Engine destroyed');if(this.lost)throw new Error('GPU device lost: '+this.lost);}
 isTensor(t){return t&&typeof t==='object'&&this.buffers.has(t);}
 buffer(value,temp=false){this.check();const data=typeof value==='number'?null:value,size=Math.max(4,Math.ceil((data?.byteLength??value)/4)*4);
  if(size>this.device.limits.maxBufferSize||size>this.device.limits.maxStorageBufferBindingSize)throw new Error(`Buffer ${size} exceeds WebGPU device limits`);
  const b=(temp&&!data?this.pool.get(size)?.pop():null)??this.device.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  if(data){const bytes=new Uint8Array(size);bytes.set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));this.device.queue.writeBuffer(b,0,bytes);}
  this.buffers.add(b);if(temp){this.temporary.add(b);this.live.add(b);}return b;
 }
 upload(name,t){if(t.dtype!=='F32'&&/embed|codebook/.test(name)){t={dtype:'F32',shape:t.shape,bytes:new Uint8Array(decodeTensor(t).buffer)};}
  this.w[name]={buffer:this.buffer(t.bytes),dtype:t.dtype,shape:t.shape,scales:t.scales?this.buffer(t.scales):this.zero};}
 async pipeline(code,count){let promise=this.pipelines.get(code);if(!promise){promise=(async()=>{const module=this.device.createShaderModule({code});const c=await module.getCompilationInfo(),errors=c.messages.filter(m=>m.type==='error');if(errors.length)throw new Error(errors.map(e=>e.message).join('\n')+'\n'+code);
  const layout=this.device.createBindGroupLayout({entries:Array.from({length:count},(_,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===count-1?'storage':'read-only-storage'}}))});
  return {layout,pipeline:await this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:'main'}})};})();this.pipelines.set(code,promise);}return promise;}
 async dispatch(code,inputs,n,groups=[Math.ceil(n/64)]){const y=this.buffer(n*4,true),{layout,pipeline}=await this.pipeline(code,inputs.length+1);
  const bind=this.device.createBindGroup({layout,entries:[...inputs,y].map((buffer,binding)=>({binding,resource:{buffer}}))});const pass=this.encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bind);if(groups.length===1&&groups[0]>65535)groups=[65535,Math.ceil(groups[0]/65535)];pass.dispatchWorkgroups(...groups);pass.end();return y;}
 elem(n,body,inputs,extra=''){return this.dispatch(S.kernel(n,body,inputs.map(()=> 'f32'),extra),inputs,n);}
 zeros(n){return this.buffer(new Float32Array(n),true);}
 linear(x,p,R,K,N){const w=this.w[p+'.weight'],bias=this.w[p+'.bias'];if(!w||w.shape.toString()!==[N,K].toString())throw new Error(`Missing/mismatched weight ${p} ${N},${K}`);return this.dispatch(S.matmulSource(R,N,K,w.dtype,!!bias,this.useF16),[x,w.buffer,bias?.buffer??this.zero,w.scales],R*N,[Math.ceil(N/16),Math.ceil(R/16)]);}
 norm(x,p,R,D,opts={}){return this.dispatch(S.normSource(R,D,{...opts,bias:!!this.w[p+'.bias']}),[x,this.w[p+'.weight'].buffer,this.w[p+'.bias']?.buffer??this.zero],R*D,[R]);}
 add(x,y,n,scale=1){return this.elem(n,`y[i]=a0[i]+a1[i]/${Number(scale).toPrecision(17)};`,[x,y]);}
 gelu(x,n){return this.elem(n,'let z=a0[i];y[i]=z*0.5*(1.0+erf_(z*0.7071067811865476));',[x],S.erfWGSL);}
 swiglu(x,R,F){return this.elem(R*F,`let r=i/${F}u;let j=i%${F}u;let v=a0[r*${2*F}u+j];y[i]=(v/(1.0+exp(-v)))*a0[r*${2*F}u+${F}u+j];`,[x]);}
 rope(x,L,D,H,B=1){let code=S.ropeSource(L*B,D,H);if(B>1)code=code.replace(`let p=i/${D}u;`,`let p=(i/${D}u)%${L}u;`);return this.dispatch(code,[x],L*B*D,[Math.ceil(L*B*D/256)]);}
 async attention(q,k,v,L,D,H,seq,B=1){if(B!==1)throw new Error('Plain attention batch is single protein');
  let code=S.scoresSource(L,D,H).replace(S.decl(2,'y','f32',true),S.decl(2,'sid')+S.decl(3,'y','f32',true)).replace(`y[i]=s*${1/Math.sqrt(D/H)};`,`y[i]=select(-3.402823e38,s*${1/Math.sqrt(D/H)},sid[r]==sid[col]);`);
  const scores=await this.dispatch(code,[q,k,seq],H*L*L,[Math.ceil(H*L*L/64)]),prob=await this.dispatch(S.softmaxSource(L,H),[scores],H*L*L,[H*L]);
  return this.dispatch(S.contextSource(L,D,H,{stride:3*D,offset:2*D}),[prob,v],L*D,[Math.ceil(L*D/64)]);}
 frames(coords,L){return this.dispatch(S.frameSource(L),[coords],L*13,[Math.ceil(L/64)]);}
 knn(coords,f,seq,mask,L,K){return this.dispatch(S.knnSource(L,K),[coords,f,seq,mask],L*K,[Math.ceil(L/64)]);}
 gather(x,ids,R,D){return this.elem(R*D,`y[i]=a0[u32(a1[i/${D}u])*${D}u+i%${D}u];`,[x,ids]);}
 embeddingLookup(ids,name,R,D){return this.gather(this.w[name].buffer,ids,R,D);}
 async geometric(z,frames,seq,chain,L,H,B,p,zero){const vec=await this.dispatch(S.geomRotateSource(B*L,H),[z,frames],B*L*H*15),s=await this.dispatch(S.geomScoresSource(L,H,B),[vec,frames,seq,chain,this.w[p+'.distance_scale_per_head'].buffer,this.w[p+'.rotation_scale_per_head'].buffer],B*H*L*L);
  const prob=await this.dispatch(S.softmaxSource(L,B*H),[s],B*H*L*L,[B*H*L]);return this.dispatch(S.geomContextSource(L,H,B,zero),[prob,vec,frames],B*L*H*3);}
 relativeEmbedding(edges,res,L,K){return this.elem(L*K*1024,`let r=i/1024u;let j=r%${K}u;let q=u32(a0[(r/${K}u)*${K}u]);let k=u32(a0[r]);let idx=u32(clamp(a1[k]-a1[q],-32.0,32.0)+33.0);y[i]=a2[idx*1024u+i%1024u];`,[edges,res,this.w['encoder.relative_positional_embedding.embedding.weight'].buffer]);}
 selectLocal(x,frames,L,K,D){return this.elem(L*D,`let r=i/${D}u;y[i]=select(0.0,a0[r*${K*D}u+i%${D}u],a1[r*13u+12u]>0.0);`,[x,frames]);}
 async codebook(z,L){const w=this.w['encoder.codebook.embeddings'];const dots=await this.dispatch(S.matmulSource(L,4096,128,'F32'),[z,w.buffer,this.zero,this.zero],L*4096,[256,Math.ceil(L/16)]);
  return this.elem(L,`var best=3.402823e38;var idx=0u;var zn=0.0;for(var d=0u;d<128u;d++){zn+=a0[i*128u+d]*a0[i*128u+d];}for(var k=0u;k<4096u;k++){var wn=0.0;for(var d=0u;d<128u;d++){let v=a1[k*128u+d];wn+=v*v;}let dist=zn-2.0*a2[i*4096u+k]+wn;if(dist<best){best=dist;idx=k;}}y[i]=f32(idx);`,[z,w.buffer,dots]);}
 structureSpecials(t,seq,L){return this.elem(L,`let s=a1[i];var v=a0[i];if(s==0.0){v=4098.0;}if(s==2.0){v=4097.0;}if(s==1.0){v=4099.0;}if(s==31.0){v=4100.0;}y[i]=v;`,[t,seq]);}
 async embedding(input){const {L}=input,D=1536;let x=await this.embeddingLookup(input.sequence_tokens,'esm.encoder.sequence_embed.weight',L,D);
  for(const [track,p] of [['average_plddt','plddt_projection'],['per_res_plddt','structure_per_res_plddt_projection']]){const rbf=await this.elem(L*16,'let v=(a0[i/16u]-f32(i%16u)/15.0)*16.0;y[i]=exp(-v*v);',[input[track]]);x=await this.add(x,await this.linear(rbf,'esm.encoder.'+p,L,16,D),L*D);}
  for(const [track,p] of [['structure_tokens','structure_tokens_embed'],['ss8_tokens','ss8_embed'],['sasa_tokens','sasa_embed']])x=await this.add(x,await this.embeddingLookup(input[track],'esm.encoder.'+p+'.weight',L,D),L*D);
  let f=this.zeros(L*D);for(let h=0;h<8;h++)f=await this.elem(L*D,`let d=i%1536u;var v=a0[i];if(d/192u==${h}u){v=a2[u32(a1[(i/1536u)*8u+${h}u])*192u+d%192u];}y[i]=v;`,[f,input.function_tokens,this.w[`esm.encoder.function_embed.${h}.weight`].buffer]);
  x=await this.add(x,f,L*D);const r=await this.elem(L*D,'let r=i/1536u;let d=i%1536u;var v=0.0;for(var j=0u;j<16u;j++){let t=u32(a0[r*16u+j]);if(t!=0u){v+=a1[t*1536u+d];}}y[i]=v;',[input.residue_annotation_tokens,this.w['esm.encoder.residue_embed.weight'].buffer]);return this.add(x,r,L*D);}
 backbone(proj,L){return this.dispatch(S.backboneSource(L),[proj],L*9,[Math.ceil(L/64)]);}
 affine(proj,L){let code=S.backboneSource(L).replace('y[i*9u+d]=n[d];y[i*9u+3u+d]=trans[d];y[i*9u+6u+d]=c[d];','for(var k=0u;k<3u;k++){y[i*12u+d*3u+k]=rot[k][d];}y[i*12u+9u+d]=trans[d];');return this.dispatch(code,[proj],L*12,[Math.ceil(L/64)]);}
 async decoderConfidence(logits,tokens,L){const sliced=await this.elem(L*L*64,`let r=i/64u;var v=a0[r*224u+160u+i%64u];if(a1[r/${L}u]>=4096.0||a1[r%${L}u]>=4096.0){v=-3.4028234663852886e38;}y[i]=v;`,[logits,tokens]);
 const probs=await this.dispatch(S.softmaxSource(64,L*L),[sliced],L*L*64,[L*L]);
 const pae=await this.elem(L*L,'var s=0.0;for(var d=0u;d<64u;d++){s+=a0[i*64u+d]*(0.25+f32(d)*0.5);}y[i]=s;',[probs]);
 const rows=await this.elem(L,`var count=0.0;for(var j=0u;j<${L}u;j++){if(a1[j]<4096.0){count+=1.0;}}let d0=1.24*pow(max(count,19.0)-15.0,1.0/3.0)-1.8;var value=0.0;if(a1[i]<4096.0){for(var j=0u;j<${L}u;j++){if(a1[j]<4096.0){for(var d=0u;d<64u;d++){let b=(0.25+f32(d)*0.5)/d0;value+=a0[(i*${L}u+j)*64u+d]/(1.0+b*b);}}}}y[i]=value/(count+1e-10);`,[probs,tokens]);
 const ptm=await this.elem(1,`var m=0.0;for(var r=0u;r<${L}u;r++){m=max(m,a0[r]);}y[0]=m;`,[rows]);return {pae,ptm};}
 async expectation(x,R,D,start,step){const p=await this.dispatch(S.softmaxSource(D,R),[x],R*D,[R]);return this.elem(R,`var v=0.0;for(var j=0u;j<${D}u;j++){v+=a0[i*${D}u+j]*(${start}+f32(j)*${step});}y[i]=v;`,[p]);}
 pairFeatures(x,L){return this.elem(L*L*128,`let d=i%128u;let c=(i/128u)%${L}u;let r=i/${L*128}u;let q=a0[c*128u+d%64u];let k=a0[r*128u+64u+d%64u];y[i]=select(q*k,q-k,d>=64u);`,[x]);}
 releaseExcept(keep){const k=new Set([...keep,...Object.values(this.traces).map(t=>t.buffer)]);for(const b of this.live)if(!k.has(b)){if(!this.pool.has(b.size))this.pool.set(b.size,[]);this.pool.get(b.size).push(b);this.live.delete(b);}}
 save(name,buffer,n){if(this.traceEnabled)this.traces[name]={buffer,n};}
 async run(fn,{signal,returnIntermediates=false}={}){this.check();if(this.busy)throw new Error('Inference already running');signal?.throwIfAborted();this.busy=true;this.temporary=new Set();this.live=new Set();this.pool=new Map();this.traces={};this.traceEnabled=returnIntermediates;this.encoder=this.device.createCommandEncoder();this.device.pushErrorScope('validation');
  try{const records=await fn();signal?.throwIfAborted();const all={...Object.fromEntries(Object.entries(this.traces).map(([k,v])=>['trace:'+k,v])),...records};const size=Object.values(all).reduce((s,t)=>s+t.n*4,0);
   const readback=this.device.createBuffer({size:Math.max(4,size),usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});this.buffers.add(readback);this.temporary.add(readback);let off=0;
   for(const t of Object.values(all)){t.offset=off;this.encoder.copyBufferToBuffer(t.buffer,0,readback,off,t.n*4);off+=t.n*4;}
   this.device.queue.submit([this.encoder.finish()]);await readback.mapAsync(GPUMapMode.READ);const mapped=readback.getMappedRange(),out={},traces={};for(const [k,t] of Object.entries(all)){const values=new Float32Array(mapped.slice(t.offset,t.offset+t.n*4));if(k.startsWith('trace:'))traces[k.slice(6)]=values;else out[k]=values;}readback.unmap();signal?.throwIfAborted();
   for(const [k,v] of Object.entries(out))for(const n of v)if(!Number.isFinite(n))throw new Error('Nonfinite output: '+k);
   if(returnIntermediates)out.intermediates=traces;return out;
  }finally{const error=await this.device.popErrorScope();for(const t of this.temporary){t.destroy();this.buffers.delete(t);}this.temporary.clear();this.live.clear();this.pool.clear();this.traces={};this.busy=false;if(error)throw new Error('WebGPU validation: '+error.message);}}
 memoryUsage(){return {bufferCount:this.buffers.size,totalBytes:[...this.buffers].reduce((s,b)=>s+b.size,0),transientBuffers:this.temporary?.size??0};}
 destroy(){if(this.destroyed)return;this.destroyed=true;for(const b of this.buffers)b.destroy();this.buffers.clear();this.w={};this.pipelines.clear();if(this.ownsDevice)this.device.destroy();}
}
