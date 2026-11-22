import {matmul,element,layerNorm,vectorNorm,scores,softmax} from './shaders.js';
import {nodeGeometry,neighbors,edgeGeometry} from './geometry.js';
export class GPUOps {
 constructor(device,{ownsDevice=false,useF16=false,onProgress}={}){this.device=device;this.ownsDevice=ownsDevice;this.useF16=useF16&&device.features.has('shader-f16');this.onProgress=onProgress;this.weights={};this.buffers=new Set();this.pipelines=new Map();this.temporary=new Set();this.live=new Set();this.pool=new Map();this.destroyed=false;this.lost=null;device.lost.then(info=>{this.lost=info.message||'WebGPU device lost';});this.zero=this.allocate(4);}
 allocate(size,temporary=false,data){
  size=Math.max(4,Math.ceil(size/4)*4);if(this.destroyed)throw new Error('Engine destroyed');if(size>this.device.limits.maxBufferSize||size>this.device.limits.maxStorageBufferBindingSize)throw new Error(`Buffer of ${size} bytes exceeds WebGPU maxBufferSize/maxStorageBufferBindingSize`);
  const buffer=(temporary&&!data?this.pool.get(size)?.pop():null)??this.device.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  if(data){const bytes=new Uint8Array(size);bytes.set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));this.device.queue.writeBuffer(buffer,0,bytes);}
  this.buffers.add(buffer);if(temporary){this.temporary.add(buffer);this.live.add(buffer);}return buffer;
 }
 tensor(data,shape){return {buffer:this.allocate(data.byteLength,true,data),shape,dtype:data instanceof Int32Array?'I32':'F32'};}
 async upload(name,t){this.weights[name]={buffer:this.allocate(t.bytes.byteLength,false,t.bytes),shape:t.shape,dtype:t.dtype,scales:t.scales?this.allocate(t.scales.byteLength,false,t.scales):this.zero};}
 admit(N,M){const E=N*Math.min(30,N),bytes=4*Math.max(E*2593,N*4096,8*Math.max(N*N,M*M,M*N));if(bytes>this.device.limits.maxBufferSize||bytes>this.device.limits.maxStorageBufferBindingSize)throw new Error(`Inference needs a ${bytes}-byte scratch buffer; exceeds WebGPU maxBufferSize/maxStorageBufferBindingSize`);}
 begin(){if(this.destroyed)throw new Error('Engine destroyed');if(this.lost)throw new Error(this.lost);this.encoder=this.device.createCommandEncoder();this.device.pushErrorScope('validation');}
 async finish(){const error=await this.device.popErrorScope();for(const b of this.temporary){b.destroy();this.buffers.delete(b);}this.temporary.clear();this.live.clear();this.pool.clear();this.encoder=null;if(error)throw new Error(`WebGPU validation: ${error.message}`);}
 keep(tensors){const keep=new Set(tensors.map(t=>t.buffer));for(const b of this.live)if(!keep.has(b)){if(!this.pool.has(b.size))this.pool.set(b.size,[]);this.pool.get(b.size).push(b);this.live.delete(b);}}
 async checkpoint(){if(this.destroyed)throw new Error('Engine destroyed');if(this.lost)throw new Error(this.lost);this.onProgress?.({phase:'inference'});await new Promise(r=>setTimeout(r,0));}
 async pipeline(code,count){let p=this.pipelines.get(code);if(!p){p=(async()=>{const module=this.device.createShaderModule({code});const info=await module.getCompilationInfo(),errors=info.messages.filter(m=>m.type==='error');if(errors.length)throw new Error(errors.map(e=>`${e.lineNum}:${e.linePos} ${e.message}`).join('\n'));
 const layout=this.device.createBindGroupLayout({entries:Array.from({length:count},(_,binding)=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===count-1?'storage':'read-only-storage'}}))});
 return {layout,pipeline:await this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:'main'}})};})();this.pipelines.set(code,p);}return p;}
 async dispatch(code,inputs,shape,groups){const n=shape.reduce((a,b)=>a*b,1),buffer=this.allocate(n*4,true),{layout,pipeline}=await this.pipeline(code,inputs.length+1),buffers=[...inputs.map(x=>x.buffer??x),buffer];
 const bind=this.device.createBindGroup({layout,entries:buffers.map((b,binding)=>({binding,resource:{buffer:b}}))}),pass=this.encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(...groups);pass.end();return {buffer,shape,dtype:'F32'};}
 elem(shape,code,inputs){const n=shape.reduce((a,b)=>a*b,1),groups=Math.ceil(n/256);return this.dispatch(element(n,code,inputs.map(x=>x.dtype==='I32'?'i32':'f32')),inputs,shape,[Math.min(groups,65535),Math.ceil(groups/65535)]);}
 linear(x,p){const w=this.weights[p+'.weight'],b=this.weights[p+'.bias'];if(!w)throw new Error(`Missing weight ${p}`);const [O,I]=w.shape,M=x.shape.reduce((a,b)=>a*b,1)/I;return this.dispatch(matmul(M,O,I,w.dtype,!!b,this.useF16),[x,w,b??this.zero,w.scales],[M,O],[Math.ceil(O/16),Math.ceil(M/16)]);}
 add(a,b){return this.elem(a.shape,'y[i]=x0[i]+x1[i];',[a,b]);}
 relu(x){return this.elem(x.shape,'y[i]=max(x0[i],0.0);',[x]);}
 async concat(xs){const M=xs[0].shape[0],widths=xs.map(x=>x.shape[1]),D=widths.reduce((a,b)=>a+b,0);let code=`let r=i/${D}u;let j=i%${D}u;`,offset=0;for(let k=0;k<xs.length;k++){code+=`${k?'else ':''}if(j<${offset+widths[k]}u){y[i]=x${k}[r*${widths[k]}u+j-${offset}u];}`;offset+=widths[k];}return this.elem([M,D],code,xs);}
 slice(x,start,width){const [M,D]=x.shape;return this.elem([M,width],`y[i]=x0[(i/${width}u)*${D}u+${start}u+i%${width}u];`,[x]);}
 norm(x,p,dihedral=false){const [M,D]=x.shape;return this.dispatch(layerNorm(M,D,dihedral),[x,this.weights[p+(dihedral?'.gain':'.weight')],this.weights[p+'.bias']],x.shape,[M]);}
 transposeVectors(x,reverse=false){const M=reverse?x.shape[0]/3:x.shape[0],V=reverse?x.shape[1]:x.shape[1]/3;return this.elem(reverse?[M,V*3]:[M*3,V],reverse?`let r=i/${V*3}u;let v=(i/3u)%${V}u;let k=i%3u;y[i]=x0[(r*3u+k)*${V}u+v];`:`let r=i/${V*3}u;let k=(i/${V}u)%3u;let v=i%${V}u;y[i]=x0[(r*${V}u+v)*3u+k];`,[x]);}
 vectorLength(x){return this.elem([x.shape[0],x.shape[1]/3],'let v=vec3<f32>(x0[i*3u],x0[i*3u+1u],x0[i*3u+2u]);y[i]=sqrt(dot(v,v)+0.00000001);',[x]);}
 vectorGate(v,g){return this.elem(v.shape,'y[i]=x0[i]/(1.0+exp(-x1[i/3u]));',[v,g]);}
 vectorNorm(x){return this.dispatch(vectorNorm(...x.shape),[x],x.shape,[Math.ceil(x.shape[0]/64)]);}
 rotate(x,f){const D=x.shape[1];return this.elem(x.shape,`let r=i/${D}u;let j=(i%${D}u)/3u;let k=i%3u;let v=vec3<f32>(x0[r*${D}u+j*3u],x0[r*${D}u+j*3u+1u],x0[r*${D}u+j*3u+2u]);let a=r*9u+k*3u;y[i]=dot(v,vec3<f32>(x1[a],x1[a+1u],x1[a+2u]));`,[x,f]);}
 gather(x,index,which){const E=index.shape[0],D=x.shape[1];return this.elem([E,D],`let r=x1[(i/${D}u)*2u+${which}u];var v=0.0;if(r>=0){v=x0[u32(r)*${D}u+i%${D}u];}y[i]=v;`,[x,index]);}
 aggregate(x,index,N){const [E,D]=x.shape;return this.elem([N,D],`let r=i/${D}u;let j=i%${D}u;var sum=0.0;var count=0.0;for(var e=0u;e<${E}u;e++){if(x1[e*2u+1u]==i32(r)){sum+=x0[e*${D}u+j];count+=1.0;}}y[i]=sum/max(count,1.0);`,[x,index]);}
 embed(tokens,p){const w=this.weights[p+'.weight'],D=w.shape[1],M=tokens.shape[0];if(w.dtype!=='F32')throw new Error('Embeddings must remain F32');return this.elem([M,D],`y[i]=x1[u32(x0[i/${D}u])*${D}u+i%${D}u]*${Math.sqrt(D)};`,[tokens,w]);}
 position(x,pad,zeroPad=false){const [M,D]=x.shape;return this.elem(x.shape,`let r=i/${D}u;let j=i%${D}u;let a=f32(r+2u)*exp(f32(j%${D/2}u)*${-Math.log(10000)/(D/2-1)});var pos=select(cos(a),sin(a),j<${D/2}u);if(x1[r]!=0){pos=0.0;}y[i]=${zeroPad?'select(x0[i]+pos,0.0,x1[r]!=0)':'x0[i]+pos'};`,[x,pad]);}
 async attention(q,k,v,padding,causal=false){const M=q.shape[0],S=k.shape[0],D=q.shape[1],groups=Math.ceil(8*M*S/256);
 const a=await this.dispatch(scores(M,S,D,causal),[q,k,padding],[8*M,S],[Math.min(groups,65535),Math.ceil(groups/65535)]),p=await this.dispatch(softmax(8*M,S),[a],[8*M,S],[8*M]);
 return this.elem([M,D],`let r=i/${D}u;let d=i%${D}u;let h=d/${D/8}u;var sum=0.0;for(var j=0u;j<${S}u;j++){sum+=x0[(h*${M}u+r)*${S}u+j]*x1[j*${D}u+d];}y[i]=sum;`,[p,v]);}
 async features(input){const N=input.N,K=Math.min(30,N),E=N*K,c=this.tensor(input.coords,[N,9]),mask=this.tensor(input.coordMask,[N,1]),cf=this.tensor(input.confidence,[N,1]);
 const node=await this.dispatch(nodeGeometry(N),[c,mask,cf],[N,47],[Math.ceil(N/64)]),index=await this.dispatch(neighbors(N,K),[c,mask],[E,2],[Math.ceil(N/64)]);index.dtype='I32';
 const edge=await this.dispatch(edgeGeometry(E),[c,index],[E,37],[Math.ceil(E/64)]);
 return {nodeScalar:await this.slice(node,0,7),nodeVector:await this.slice(node,7,9),dihedrals:await this.slice(node,16,6),frames:await this.slice(node,22,9),confidenceRBF:await this.slice(node,31,16),edgeIndex:index,edgeScalar:await this.slice(edge,0,34),edgeVector:await this.slice(edge,34,3)};
 }
 async readMany(tensors){const entries=Object.entries(tensors),size=entries.reduce((n,[,t])=>n+t.shape.reduce((a,b)=>a*b,1)*4,0);if(size>this.device.limits.maxBufferSize)throw new Error('Readback exceeds maxBufferSize');
 const readback=this.device.createBuffer({size:Math.max(4,size),usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});this.buffers.add(readback);this.temporary.add(readback);let offset=0;for(const [,t]of entries){const bytes=t.shape.reduce((a,b)=>a*b,1)*4;this.encoder.copyBufferToBuffer(t.buffer,0,readback,offset,bytes);offset+=bytes;}
 this.device.queue.submit([this.encoder.finish()]);this.encoder=this.device.createCommandEncoder();this.onProgress?.({phase:'submitted'});
 await readback.mapAsync(GPUMapMode.READ);const data=readback.getMappedRange(),values={};offset=0;for(const [name,t]of entries){const bytes=t.shape.reduce((a,b)=>a*b,1)*4;values[name]=new (t.dtype==='I32'?Int32Array:Float32Array)(data.slice(offset,offset+bytes));offset+=bytes;}readback.unmap();readback.destroy();this.buffers.delete(readback);this.temporary.delete(readback);return values;
 }
 async read(t){return (await this.readMany({value:t})).value;}
 memoryUsage(){return {bufferCount:this.buffers.size,totalBytes:[...this.buffers].reduce((n,b)=>n+b.size,0),transientBuffers:this.temporary.size,transientBytes:[...this.temporary].reduce((n,b)=>n+b.size,0)};}
 destroy(){if(this.destroyed)return;this.destroyed=true;for(const b of this.buffers)b.destroy();this.buffers.clear();this.temporary.clear();this.live.clear();this.pool.clear();this.weights={};this.pipelines.clear();if(this.ownsDevice)this.device.destroy();}
}
