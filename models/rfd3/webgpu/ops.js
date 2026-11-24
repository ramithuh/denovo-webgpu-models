// Eager WGSL tensor operations. All activation arithmetic stays on the GPU.
// Scratch buffers are reused only after their last encoded use; host uploads are
// never pooled, so queue.writeBuffer cannot overwrite earlier pending commands.
export class GPUOps {
  constructor(resources) {
    this.owner=resources;this.device=resources.device;this.pipelines=new Map();this.pool=new Map();this.tensors=new Set();this.scopes=[];this.encoder=null;
  }
  begin(){if(this.encoder)throw new Error('GPU command transaction already active');this.encoder=this.device.createCommandEncoder();}
  tensor(length,host=false){
    if(!Number.isSafeInteger(length)||length<1||length*4>this.device.limits.maxStorageBufferBindingSize)throw new RangeError('Tensor exceeds the negotiated binding limit');
    const size=length*4,bucket=this.pool.get(size),buffer=!host&&bucket?.length?bucket.pop():this.owner.allocate(size,128|8|4,'RFD3 activation');
    const t={buffer,length,host,released:false};this.tensors.add(t);for(const scope of this.scopes)scope.push(t);return t;
  }
  upload(data){const t=this.tensor(data.length,true);this.device.queue.writeBuffer(t.buffer,0,Float32Array.from(data));return t;}
  free(t){if(!t||t.released||!this.tensors.delete(t))return;t.released=true;if(t.host){this.owner.release(t.buffer);return;}let b=this.pool.get(t.buffer.size);if(!b)this.pool.set(t.buffer.size,b=[]);b.push(t.buffer);}
  scope(fn){const created=[];this.scopes.push(created);try{const result=fn(),keep=new Set();const walk=v=>{if(!v||typeof v!=='object')return;if(v.buffer&&v.length!==undefined){keep.add(v);return;}for(const x of Object.values(v))walk(x);};walk(result);for(const t of created)if(!keep.has(t))this.free(t);return result;}catch(error){for(const t of created)this.free(t);throw error;}finally{this.scopes.pop();}}
  dispatch(code,buffers,groups,label='RFD3 op'){
    if(!this.encoder)throw new Error('No GPU command transaction');
    let pipeline=this.pipelines.get(code);
    if(!pipeline){const module=this.device.createShaderModule({code,label});pipeline=this.device.createComputePipeline({layout:'auto',compute:{module,entryPoint:'main'},label});this.pipelines.set(code,pipeline);}
    const group=this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:buffers.map((t,binding)=>({binding,resource:{buffer:t.buffer??t}}))});
    const pass=this.encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(...groups);pass.end();
  }
  kernel(inputs,length,body,label='RFD3 elementwise'){
    const out=this.tensor(length),num=Math.ceil(length/64),x=Math.min(65535,num),y=Math.ceil(num/x);
    const code=inputs.map((_,i)=>`@group(0) @binding(${i}) var<storage,read> a${i}:array<f32>;`).join('\n')+`
@group(0) @binding(${inputs.length}) var<storage,read_write> out:array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){let idx=gid.x+gid.y*${x*64}u;if(idx>=${length}u){return;}${body}}`;
    this.dispatch(code,[...inputs,out],[x,y],label);return out;
  }
  fill(length,value=0){return this.kernel([],length,`out[idx]=${Number(value).toFixed(12)};`);}
  add(a,b){if(a.length!==b.length)throw new Error('GPU add shape mismatch');return this.kernel([a,b],a.length,'out[idx]=a0[idx]+a1[idx];');}
  mul(a,b){return this.kernel([a,b],a.length,'out[idx]=a0[idx]*a1[idx];');}
  activation(a,kind){const expression={relu:'max(v,0.0)',sigmoid:'1.0/(1.0+exp(-v))',silu:'v/(1.0+exp(-v))'}[kind];if(!expression)throw new Error('Unknown activation');return this.kernel([a],a.length,`let v=a0[idx];out[idx]=${expression};`);}
  linear(a,w,bias,{reluInput=false}={}){
    const [n,k]=w.shape,m=a.length/k;if(!Number.isInteger(m))throw new Error('GPU linear shape mismatch');
    const out=this.tensor(m*n),dtype={F32:0,F16:1,I8:2}[w.dtype],inputs=[a,w];
    let scale='',biasCode='';if(dtype===2){inputs.push(w.scales);scale='*scales[row]';}if(bias){inputs.push(bias);biasCode='sum+=bias[col];';}
    const code=`@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> w:array<u32>;
${dtype===2?'@group(0) @binding(2) var<storage,read> scales:array<f32>;':''}
${bias?`@group(0) @binding(${inputs.length-1}) var<storage,read> bias:array<f32>;`:''}
@group(0) @binding(${inputs.length}) var<storage,read_write> out:array<f32>;
var<workgroup> a:array<f32,128>;var<workgroup> b:array<f32,128>;
fn weight(i:u32,row:u32)->f32 {${dtype===0?'return bitcast<f32>(w[i]);':dtype===1?'return unpack2x16float(w[i/2u])[i%2u];':`return f32(bitcast<i32>(((w[i/4u]>>(8u*(i%4u)))&255u)<<24u)>>24)${scale};`}}
@compute @workgroup_size(8,8) fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) id:vec3<u32>){
 let row=(g.y+g.z*65535u)*8u+id.y;let col=g.x*8u+id.x;var sum=0.0;
 for(var base=0u;base<${k}u;base+=16u){for(var j=0u;j<2u;j++){let ka=id.x+8u*j;let kb=id.y+8u*j;var av=0.0;var bv=0.0;
 if(row<${m}u&&base+ka<${k}u){av=${reluInput?`max(x[row*${k}u+base+ka],0.0)`:`x[row*${k}u+base+ka]`};}if(col<${n}u&&base+kb<${k}u){bv=weight(col*${k}u+base+kb,col);}a[id.y*16u+ka]=av;b[id.x*16u+kb]=bv;}
 workgroupBarrier();for(var j=0u;j<16u;j++){sum+=a[id.y*16u+j]*b[id.x*16u+j];}workgroupBarrier();}
 if(row<${m}u&&col<${n}u){${biasCode}out[row*${n}u+col]=sum;}}`;
    this.dispatch(code,[...inputs,out],[Math.ceil(n/8),Math.min(65535,Math.ceil(m/8)),Math.ceil(m/8/65535)],'RFD3 packed linear');return out;
  }
  fusedPosition(pos,mask,wA,bA,wB,bB,{sinusoidal=false,addTo}={}){
    const n=pos.length/3,channels=wA.shape[0],total=n*n*channels;
    const dtypeA={F32:0,F16:1,I8:2}[wA.dtype],dtypeB={F32:0,F16:1,I8:2}[wB.dtype];
    const inputs=[pos,mask,wA,wB],decls=[
      '@group(0) @binding(0) var<storage,read> pos:array<f32>;',
      '@group(0) @binding(1) var<storage,read> mask:array<f32>;',
      '@group(0) @binding(2) var<storage,read> wA:array<u32>;',
      '@group(0) @binding(3) var<storage,read> wB:array<u32>;',
    ];
    const addBuf=(buf,name)=>{const i=inputs.length;inputs.push(buf);decls.push(`@group(0) @binding(${i}) var<storage,read> ${name}:array<f32>;`);};
    if(dtypeA===2)addBuf(wA.scales,'sA');
    if(bA)addBuf(bA,'bA');
    if(dtypeB===2)addBuf(wB.scales,'sB');
    if(bB)addBuf(bB,'bB');
    if(addTo)addBuf(addTo,'addIn');
    const out=this.tensor(total),num=Math.ceil(total/64),x=Math.min(65535,num),y=Math.ceil(num/x);
    decls.push(`@group(0) @binding(${inputs.length}) var<storage,read_write> out:array<f32>;`);
    const unpack=(wName,sName,dt)=>dt===0?`return bitcast<f32>(${wName}[i]);`:dt===1?`return unpack2x16float(${wName}[i/2u])[i%2u];`:`return f32(bitcast<i32>(((${wName}[i/4u]>>(8u*(i%4u)))&255u)<<24u)>>24)*${sName}[row];`;
    const code=`${decls.join('\n')}
fn weightA(i:u32,row:u32)->f32 {${unpack('wA','sA',dtypeA)}}
fn weightB(i:u32,row:u32)->f32 {${unpack('wB','sB',dtypeB)}}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 let idx=gid.x+gid.y*${x*64}u;if(idx>=${total}u){return;}
 let pair=idx/${channels}u;let m=mask[pair];
 if(m==0.0){out[idx]=${addTo?'addIn[idx]':'0.0'};return;}
 let i=pair/${n}u;let j=pair%${n}u;let col=idx%${channels}u;
 var square=0.0;for(var c=0u;c<3u;c++){let d=pos[i*3u+c]-pos[j*3u+c];square+=d*d;}
 ${sinusoidal
   ?`let dist=sqrt(square);var sumA=0.0;for(var k=0u;k<64u;k++){let angle=dist*exp(-log(10000.0)*f32(k%32u)/32.0);sumA+=select(sin(angle),cos(angle),k>=32u)*weightA(col*64u+k,col);}`
   :`var sumA=(1.0/(1.0+max(square,1e-6)))*weightA(col,col);`}
 ${bA?'sumA+=bA[col];':''}
 var sumB=m*weightB(col,col);
 ${bB?'sumB+=bB[col];':''}
 let v=(sumA+sumB)*m;
 out[idx]=${addTo?'addIn[idx]+v':'v'};
}`;
    this.dispatch(code,[...inputs,out],[x,y],'RFD3 fused position embed');return out;
  }
  norm(a,channels,gain){
    const rows=a.length/channels,out=this.tensor(a.length);if(!Number.isInteger(rows))throw new Error('GPU RMSNorm shape mismatch');
    const code=`@group(0) @binding(0) var<storage,read> a:array<f32>;
${gain?'@group(0) @binding(1) var<storage,read> gain:array<f32>;':''}
@group(0) @binding(${gain?2:1}) var<storage,read_write> out:array<f32>;
var<workgroup> sums:array<f32,64>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) id:u32){let row=g.x+g.y*65535u;if(row>=${rows}u){return;}var s=0.0;
 for(var c=id;c<${channels}u;c+=64u){let v=a[row*${channels}u+c];s+=v*v;}sums[id]=s;workgroupBarrier();
 for(var step=32u;step>0u;step/=2u){if(id<step){sums[id]+=sums[id+step];}workgroupBarrier();}
 let scale=inverseSqrt(sums[0]/${channels}.0+0.00000011920928955078125);
 for(var c=id;c<${channels}u;c+=64u){out[row*${channels}u+c]=a[row*${channels}u+c]*scale${gain?'*gain[c]':''};}}`;
    this.dispatch(code,gain?[a,gain,out]:[a,out],[Math.min(rows,65535),Math.ceil(rows/65535)],'RFD3 RMSNorm');return out;
  }
  concat(arrays,widths,rows){let offset=0;const width=widths.reduce((a,b)=>a+b,0),clauses=widths.map((w,i)=>{const start=offset;offset+=w;return `if(c>=${start}u&&c<${offset}u){out[idx]=a${i}[row*${w}u+c-${start}u];}`;});return this.kernel(arrays,rows*width,`let row=idx/${width}u;let c=idx%${width}u;${clauses.join('')}`);}
  attention(q,k,v,{queries,keys,channels,heads,bias,mask,gate,qPreScale=false}){
    return this.scope(()=>{
      const dim=channels/heads,inputs=[q,k];let biasIndex,maskIndex;
      if(bias){biasIndex=inputs.length;inputs.push(bias);}if(mask){maskIndex=inputs.length;inputs.push(mask);}
      const scores=this.kernel(inputs,queries*heads*keys,`let key=idx%${keys}u;let head=(idx/${keys}u)%${heads}u;let query=idx/${keys*heads}u;
${mask?`if(a${maskIndex}[query*${keys}u+key]==0.0){out[idx]=-1e30;return;}`:''}
 var sum=0.0;for(var c=0u;c<${dim}u;c++){sum+=${qPreScale?`(a0[query*${channels}u+head*${dim}u+c]/sqrt(${dim}.0))`:`a0[query*${channels}u+head*${dim}u+c]`}*a1[key*${channels}u+head*${dim}u+c];}
 out[idx]=${qPreScale?'sum':`sum/sqrt(${dim}.0)`}${bias?`+a${biasIndex}[(query*${keys}u+key)*${heads}u+head]`:''}${mask?`+log(a${maskIndex}[query*${keys}u+key])`:''};`,'RFD3 attention scores');
      const prob=this.softmax(scores,keys);
      const values=[prob,v];if(gate)values.push(gate);
      return this.kernel(values,queries*channels,`let query=idx/${channels}u;let head=(idx%${channels}u)/${dim}u;let c=idx%${dim}u;var sum=0.0;
 for(var key=0u;key<${keys}u;key++){let p=a0[(query*${heads}u+head)*${keys}u+key];if(p!=0.0){sum+=p*a1[key*${channels}u+head*${dim}u+c];}}out[idx]=sum${gate?'*a2[idx]':''};`,'RFD3 attention values');
    });
  }
  sortRanks(a,width){
    const size=2**Math.ceil(Math.log2(width)),rows=a.length/width,out=this.tensor(a.length);
    const code=`@group(0) @binding(0) var<storage,read> a:array<f32>;@group(0) @binding(1) var<storage,read_write> out:array<f32>;
var<workgroup> value:array<f32,${size}>;var<workgroup> index:array<u32,${size}>;
@compute @workgroup_size(${size}) fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) id:u32){value[id]=3.402823e38;index[id]=id;workgroupBarrier();if(id<${width}u){var maximum=-3.4e38;for(var j=0u;j<${width}u;j++){maximum=max(maximum,a[g.x*${width}u+j]);}let v=a[g.x*${width}u+id];var before=0u;var lower=0u;for(var j=0u;j<${width}u;j++){let other=a[g.x*${width}u+j];if(other<maximum){lower++;if(j<id&&v<maximum){before++;}}else if(j<id&&v==maximum){before++;}}let position=before+select(0u,lower,v==maximum);value[position]=v;index[position]=id;}workgroupBarrier();
for(var span=2u;span<=${size}u;span*=2u){for(var stride=span/2u;stride>0u;stride/=2u){let other=id^stride;if(other>id){let descending=(id&span)!=0u;if((value[id]<value[other])==descending){let v=value[id];value[id]=value[other];value[other]=v;let i=index[id];index[id]=index[other];index[other]=i;}}workgroupBarrier();}}
if(index[id]<${width}u){out[g.x*${width}u+index[id]]=f32(id);}}`;
    this.dispatch(code,[a,out],[rows],'RFD3 small neighbour rank');return out;
  }
  softmax(a,channels){
    const rows=a.length/channels,out=this.tensor(a.length);
    const code=`@group(0) @binding(0) var<storage,read> a:array<f32>;@group(0) @binding(1) var<storage,read_write> out:array<f32>;
var<workgroup> tmp:array<f32,64>;var<workgroup> maximum:f32;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) id:u32){let row=g.x;var m=-1e30;
for(var c=id;c<${channels}u;c+=64u){m=max(m,a[row*${channels}u+c]);}tmp[id]=m;workgroupBarrier();
for(var s=32u;s>0u;s/=2u){if(id<s){tmp[id]=max(tmp[id],tmp[id+s]);}workgroupBarrier();}if(id==0u){maximum=tmp[0];}workgroupBarrier();
var sum=0.0;for(var c=id;c<${channels}u;c+=64u){let p=exp(a[row*${channels}u+c]-maximum);out[row*${channels}u+c]=p;sum+=p;}tmp[id]=sum;workgroupBarrier();
for(var s=32u;s>0u;s/=2u){if(id<s){tmp[id]+=tmp[id+s];}workgroupBarrier();}for(var c=id;c<${channels}u;c+=64u){out[row*${channels}u+c]/=tmp[0];}}`;
    this.dispatch(code,[a,out],[rows],'RFD3 softmax');return out;
  }
  async submit(){if(!this.encoder)return;const encoder=this.encoder;this.encoder=null;this.device.queue.submit([encoder.finish()]);await this.device.queue.onSubmittedWorkDone();}
  async read(t){const buffer=this.owner.allocate(t.length*4,1|8,'RFD3 readback');try{const enc=this.device.createCommandEncoder();enc.copyBufferToBuffer(t.buffer,0,buffer,0,t.length*4);this.device.queue.submit([enc.finish()]);await buffer.mapAsync(1);const result=new Float32Array(buffer.getMappedRange().slice(0));buffer.unmap();return result;}finally{this.owner.release(buffer);}}
  trim(){if(this.encoder)throw new Error('Cannot trim pending GPU commands');for(const bucket of this.pool.values())for(const b of bucket)this.owner.release(b);this.pool.clear();}
  destroy(){for(const t of [...this.tensors]){this.owner.release(t.buffer);t.released=true;}this.tensors.clear();for(const bucket of this.pool.values())for(const b of bucket)this.owner.release(b);this.pool.clear();this.pipelines.clear();this.encoder=null;}
}
