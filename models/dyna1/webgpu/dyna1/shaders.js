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
 let r=g.x+g.y*65535u;if(r>=${L}u){return;}var s=0.0;
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
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>) {let i=g.x+g.y*16776960u;if(i<${n}u){${expression}}}`;
}
export function gatherSource(L,D) {return decl(0,'tokens','u32')+decl(1,'w')+decl(2,'y','f32',true)+`
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>) {let i=g.x+g.y*16776960u;if(i<${L*D}u){y[i]=w[tokens[i/${D}u]*${D}u+i%${D}u];}}`;}
export function gatherAddSource(L,D) {return decl(0,'seqTok','u32')+decl(1,'structTok','u32')+decl(2,'emb')+decl(3,'delta')+decl(4,'y','f32',true)+`
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>) {let i=g.x+g.y*16776960u;if(i<${L*D}u){let r=i/${D}u;let d=i%${D}u;y[i]=emb[seqTok[r]*${D}u+d]+delta[structTok[r]*${D}u+d];}}`;}
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
 let row=g.x+g.y*65535u;if(row>=${H*L}u){return;}let off=row*${L}u;var m=-3.4028234663852886e38;
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
 let i=g.x+g.y*4194240u;if(i>=${L*D}u){return;}let r=i/${D}u;let d=i%${D}u;let h=d/${dh}u;var s=0.0;
 for(var j=0u;j<${L}u;j++){s+=p[(h*${L}u+r)*${L}u+j]*v[j*${stride}u+${offset}u+d];}y[i]=s;
}`;
}
/** ESM-3 StructureTokenEncoder geometric attention and VQ codebook shaders. */
export function geomRotateSource(R,H) {
 const n=R*H*15;
 return decl(0,'z')+decl(1,'frames')+decl(2,'y','f32',true)+`
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>) {
 let i=g.x+g.y*4194240u;if(i>=${n}u){return;}
 let r=i/${H*15}u;let v=(i%${H*15}u)/3u;let d=i%3u;let off=r*${H*15}u+v*3u;let f=r*13u+d*3u;
 var s=z[off]*frames[f]+z[off+1u]*frames[f+1u]+z[off+2u]*frames[f+2u];
 if(v>=${3*H}u){s+=frames[r*13u+9u+d];}
 y[i]=s;
}`;
}
export function geomScoresSource(K,H,B) {
 const n=B*H*K*K;
 return decl(0,'vec')+decl(1,'frames')+decl(2,'dScale')+decl(3,'rScale')+decl(4,'y','f32',true)+`
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>) {
 let i=g.x+g.y*4194240u;if(i>=${n}u){return;}
 let col=i%${K}u;let row=(i/${K}u)%${K}u;let h=(i/${K*K}u)%${H}u;let batch=i/${H*K*K}u;
 let r=batch*${K}u+row;let c=batch*${K}u+col;
 let qi=r*${H*15}u+h*3u;let ki=c*${H*15}u+${H*3}u+h*3u;
 var rot=0.0;var dis=0.0;
 for(var d=0u;d<3u;d++){rot+=vec[qi+d]*vec[ki+d];let e=vec[qi+${H*9}u+d]-vec[ki+${H*9}u+d];dis+=e*e;}
 let bias=select(-3.4028234663852886e38,1.0,frames[c*13u+12u]>0.0);
 let rs=rScale[h];let ds=dScale[h];
 let rw=select(log(1.0+exp(rs)),rs,rs>20.0);
 let dw=select(log(1.0+exp(ds)),ds,ds>20.0);
 y[i]=(rot*0.5773502691896258)*rw-(sqrt(dis)*0.5773502691896258)*dw+bias;
}`;
}
export function geomContextSource(K,H,B) {
 const n=B*K*H*3;
 return decl(0,'prob')+decl(1,'vec')+decl(2,'frames')+decl(3,'y','f32',true)+`
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>) {
 let i=g.x+g.y*4194240u;if(i>=${n}u){return;}
 let d=i%3u;let h=(i/3u)%${H}u;let row=(i/${H*3}u)%${K}u;let batch=i/${K*H*3}u;let r=batch*${K}u+row;
 var v=vec3<f32>(0.0);
 for(var col=0u;col<${K}u;col++){
  let p=prob[((batch*${H}u+h)*${K}u+row)*${K}u+col];
  let k=(batch*${K}u+col)*${H*15}u+${H*6}u+h*3u;
  v+=p*vec3<f32>(vec[k],vec[k+1u],vec[k+2u]);
 }
 y[i]=v.x*frames[r*13u+d]+v.y*frames[r*13u+3u+d]+v.z*frames[r*13u+6u+d];
}`;
}
export function selectLocalSource(B,K,D) {
 return decl(0,'x')+decl(1,'frames')+decl(2,'y','f32',true)+`
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>) {
 let i=g.x+g.y*16776960u;if(i>=${B*D}u){return;}
 let r=i/${D}u;
 y[i]=select(0.0,x[r*${K*D}u+i%${D}u],frames[r*13u+12u]>0.0);
}`;
}
export function codebookArgminSource(L,codes=4096,dim=128) {
 return decl(0,'z')+decl(1,'dots')+decl(2,'norms')+decl(3,'y','u32',true)+`
var<workgroup> bestD:array<f32,256>;
var<workgroup> bestI:array<u32,256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) t:u32,@builtin(workgroup_id) g:vec3<u32>) {
 let r=g.x;if(r>=${L}u){return;}
 var zn=0.0;for(var d=0u;d<${dim}u;d++){let v=z[r*${dim}u+d];zn+=v*v;}
 var bd=3.4028234663852886e38;var bi=0u;
 for(var k=t;k<${codes}u;k+=256u){
  let dist=zn-2.0*dots[r*${codes}u+k]+norms[k];
  if(dist<bd||(dist==bd&&k<bi)){bd=dist;bi=k;}
 }
 bestD[t]=bd;bestI[t]=bi;workgroupBarrier();
 for(var n=128u;n>0u;n/=2u){
  if(t<n){
   let od=bestD[t+n];let oi=bestI[t+n];
   if(od<bestD[t]||(od==bestD[t]&&oi<bestI[t])){bestD[t]=od;bestI[t]=oi;}
  }
  workgroupBarrier();
 }
 if(t==0u){y[r]=bestI[0];}
}`;
}
