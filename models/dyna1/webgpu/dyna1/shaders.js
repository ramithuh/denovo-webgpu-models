/** WGSL generators: FP32 accumulation for every storage precision. */
const decl=(i,name,type='f32',rw=false)=>`@group(0) @binding(${i}) var<storage,${rw?'read_write':'read'}> ${name}: array<${type}>;\n`;
export function matmulSource(M,N,K,dtype='F32',hasBias=false,useF16=false) {
  const native=dtype==='F16'&&useF16;
  const get=dtype==='F32'?'return w[i];':dtype==='F16'?(native?'return f32(w[i]);':'return unpack2x16float(w[i/2u])[i%2u];'):'let q=i32((w[i/4u] >> ((i%4u)*8u)) & 255u); return f32(select(q,q-256,q>=128))*scale[row];';
  return `${native?'enable f16;':''}
${decl(0,'a')}${decl(1,'w',dtype==='F32'?'f32':native?'f16':'u32')}${decl(2,'bias')}${decl(3,'scale')}${decl(4,'y','f32',true)}
fn weight(i:u32,row:u32)->f32 {${get}}
var<workgroup> at: array<f32,256>;
var<workgroup> bt: array<f32,256>;
@compute @workgroup_size(16,16) fn main(@builtin(local_invocation_id) tid:vec3<u32>,@builtin(workgroup_id) gid:vec3<u32>) {
 let row=gid.y*16u+tid.y;let col=gid.x*16u+tid.x;let idx=tid.y*16u+tid.x;
 var sum=0.0;
 for(var base=0u;base<${K}u;base+=16u) {
  var av=0.0;var bv=0.0;
  if(row<${M}u && base+tid.x<${K}u) {av=a[row*${K}u+base+tid.x];}
  if(col<${N}u && base+tid.y<${K}u) {bv=weight(col*${K}u+base+tid.y,col);}
  at[idx]=av;bt[idx]=bv;workgroupBarrier();
  for(var k=0u;k<16u;k++) {sum=sum+at[tid.y*16u+k]*bt[k*16u+tid.x];}
  workgroupBarrier();
 }
 if(row<${M}u && col<${N}u) {y[row*${N}u+col]=sum${hasBias?'+bias[col]':''};}
}`;
}
export function normSource(L,D,{stride=D,offset=0,eps=1e-5,bias=true}={}) {
 return `${decl(0,'x')}${decl(1,'w')}${decl(2,'b')}${decl(3,'y','f32',true)}
var<workgroup> temp:array<f32,256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) t:u32,@builtin(workgroup_id) g:vec3<u32>) {
 let r=g.x;var s=0.0;
 for(var j=t;j<${D}u;j+=256u) {s+=x[r*${stride}u+${offset}u+j];}temp[t]=s;workgroupBarrier();
 for(var n=128u;n>0u;n/=2u) {if(t<n){temp[t]+=temp[t+n];}workgroupBarrier();}
 let mean=temp[0]/${D}.0;var v=0.0;
 for(var j=t;j<${D}u;j+=256u) {let z=x[r*${stride}u+${offset}u+j]-mean;v+=z*z;}
 workgroupBarrier();temp[t]=v;workgroupBarrier();
 for(var n=128u;n>0u;n/=2u) {if(t<n){temp[t]+=temp[t+n];}workgroupBarrier();}
 let inv=inverseSqrt(temp[0]/${D}.0+${eps});
 for(var j=t;j<${D}u;j+=256u) {y[r*${D}u+j]=(x[r*${stride}u+${offset}u+j]-mean)*inv*w[j]${bias?'+b[j]':''};}
}`;
}
export function elementSource(n,expression,inputs=1) {
 return Array.from({length:inputs},(_,i)=>decl(i,['x','b','c'][i])).join('')+decl(inputs,'y','f32',true)+`
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>) {let i=g.x;if(i<${n}u){${expression}}}`;
}
export function gatherSource(L,D) {return decl(0,'tokens','u32')+decl(1,'w')+decl(2,'y','f32',true)+`
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>) {let i=g.x;if(i<${L*D}u){y[i]=w[tokens[i/${D}u]*${D}u+i%${D}u];}}`;}
export function ropeSource(L,D,H) {
 const dh=D/H,half=dh/2;
 return elementSource(L*D,`let p=i/${D}u;let d=i%${dh}u;let j=d%${half}u;let angle=f32(p)*pow(10000.0,-f32(j)*${2/dh});let other=select(i+${half}u,i-${half}u,d>=${half}u);let sign=select(-1.0,1.0,d>=${half}u);y[i]=x[i]*cos(angle)+sign*x[other]*sin(angle);`);
}
export function scoresSource(L,D,H,{packedQkv=false}={}) {
 const dh=D/H;
 const rowStride=packedQkv?3*D:D;
 const keyOffset=packedQkv?D:0;
 return decl(0,'q')+decl(1,'k')+decl(2,'y','f32',true)+`
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){let i=g.x+g.y*4194240u;if(i>=${H*L*L}u){return;}let h=i/${L*L}u;let r=(i/${L}u)%${L}u;let col=i%${L}u;var s=0.0;for(var j=0u;j<${dh}u;j++){s+=q[r*${rowStride}u+h*${dh}u+j]*k[col*${rowStride}u+${keyOffset}u+h*${dh}u+j];}y[i]=s*${1/Math.sqrt(dh)};}`;
}
export function softmaxSource(L,H) {
 return decl(0,'x')+decl(1,'y','f32',true)+`
var<workgroup> t:array<f32,256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) j:u32,@builtin(workgroup_id) g:vec3<u32>) {
 let off=g.x*${L}u;var m=-3.402823e38;
 for(var k=j;k<${L}u;k+=256u){m=max(m,x[off+k]);}t[j]=m;workgroupBarrier();
 for(var n=128u;n>0u;n/=2u){if(j<n){t[j]=max(t[j],t[j+n]);}workgroupBarrier();}m=t[0];
 var s=0.0;for(var k=j;k<${L}u;k+=256u){s+=exp(x[off+k]-m);}workgroupBarrier();t[j]=s;workgroupBarrier();
 for(var n=128u;n>0u;n/=2u){if(j<n){t[j]+=t[j+n];}workgroupBarrier();}let den=t[0];
 for(var k=j;k<${L}u;k+=256u){y[off+k]=exp(x[off+k]-m)/den;}
}`;
}
export function contextSource(L,D,H,{stride=D,offset=0}={}) {
 const dh=D/H;
 return decl(0,'p')+decl(1,'v')+decl(2,'y','f32',true)+`
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>) {
 let i=g.x;if(i>=${L*D}u){return;}let r=i/${D}u;let d=i%${D}u;let h=d/${dh}u;var s=0.0;
 for(var j=0u;j<${L}u;j++){s+=p[(h*${L}u+r)*${L}u+j]*v[j*${stride}u+${offset}u+d];}y[i]=s;
}`;
}
