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
 let i=g.x;if(i>=${L*D}u){return;}let r=i/${D}u;let d=i%${D}u;let h=d/${dh}u;var s=0.0;
 for(var j=0u;j<${L}u;j++){s+=p[(h*${L}u+r)*${L}u+j]*v[j*${stride}u+${offset}u+d];}y[i]=s;
}`;
}

export {decl};
export function kernel(n,body,types=['f32'],extra=''){
 return types.map((t,i)=>decl(i,'a'+i,t)).join('')+decl(types.length,'y','f32',true)+extra+`
 @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x+gid.y*4194240u;if(i>=${n}u){return;}${body}}`;
}
export const erfWGSL=`fn erf_(v:f32)->f32{let x=abs(v);let t=1.0/(1.0+0.3275911*x);let p=(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t;return sign(v)*(1.0-p*exp(-x*x));}`;
export const frameHelpers=`
fn gs(x:vec3<f32>,v:vec3<f32>)->mat3x3<f32>{let e0=x/sqrt(dot(x,x)+1e-10);let z=v-e0*dot(e0,v);let e1=z/sqrt(dot(z,z)+1e-10);return mat3x3<f32>(e0,e1,cross(e0,e1));}
fn xyz(k:u32)->vec3<f32>{return vec3<f32>(a0[k],a0[k+1u],a0[k+2u]);}
fn valid(r:u32)->bool{var ok=true;for(var j=0u;j<9u;j++){let v=a0[r*9u+j];ok=ok&&(abs(v)<3.402823e38)&&(v<1e6);}return ok;}
`;
export function frameSource(L){return kernel(L,`
 var n=xyz(i*9u);var ca=xyz(i*9u+3u);var c=xyz(i*9u+6u);let ok=valid(i);var any_=ok;
 if(!ok){n=vec3<f32>(0.0);ca=n;c=n;var count=0.0;for(var j=0u;j<${L}u;j++){if(valid(j)){n+=xyz(j*9u);ca+=xyz(j*9u+3u);c+=xyz(j*9u+6u);count+=1.0;}}n/=count+1e-8;ca/=count+1e-8;c/=count+1e-8;any_=count>0.0;}
 var rot=gs(ca-c,n-ca);if(!any_){rot=mat3x3<f32>(vec3<f32>(1.0,0.0,0.0),vec3<f32>(0.0,1.0,0.0),vec3<f32>(0.0,0.0,1.0));}
 for(var r=0u;r<3u;r++){for(var k=0u;k<3u;k++){y[i*13u+r*3u+k]=rot[k][r];}y[i*13u+9u+r]=ca[r];}y[i*13u+12u]=select(0.0,1.0,ok);
 `,['f32'],frameHelpers);}
export function geomRotateSource(R,H){return kernel(R*H*15,`
 let r=i/${H*15}u;let v=(i%${H*15}u)/3u;let d=i%3u;let off=r*${H*15}u+v*3u;let f=r*13u+d*3u;
 var s=a0[off]*a1[f]+a0[off+1u]*a1[f+1u]+a0[off+2u]*a1[f+2u];if(v>=${3*H}u){s+=a1[r*13u+9u+d];}y[i]=s;`,['f32','f32']);}
export function geomScoresSource(L,H,B){return kernel(B*H*L*L,`
 let col=i%${L}u;let row=(i/${L}u)%${L}u;let h=(i/${L*L}u)%${H}u;let batch=i/${H*L*L}u;let r=batch*${L}u+row;let c=batch*${L}u+col;
 let qi=r*${H*15}u+h*3u;let ki=c*${H*15}u+${H*3}u+h*3u;var rot=0.0;var dis=0.0;
 for(var d=0u;d<3u;d++){rot+=a0[qi+d]*a0[ki+d];let e=a0[qi+${H*9}u+d]-a0[ki+${H*9}u+d];dis+=e*e;}
 var bias=select(0.0,1.0,a2[r]==a2[c]);if(a1[c*13u+12u]==0.0||a3[r]!=a3[c]){bias=-3.4028234663852886e38;}
 let rw=select(log(1.0+exp(a5[h])),a5[h],a5[h]>20.0);let dw=select(log(1.0+exp(a4[h])),a4[h],a4[h]>20.0);
 y[i]=(rot*0.5773502691896258)*rw-(sqrt(dis)*0.5773502691896258)*dw+bias;`,['f32','f32','f32','f32','f32','f32']);}
export function geomContextSource(L,H,B,zero){return kernel(B*L*H*3,`
 let d=i%3u;let h=(i/3u)%${H}u;let row=(i/${H*3}u)%${L}u;let batch=i/${L*H*3}u;let r=batch*${L}u+row;
 var v=vec3<f32>(0.0);for(var col=0u;col<${L}u;col++){let p=a0[((batch*${H}u+h)*${L}u+row)*${L}u+col];let k=(batch*${L}u+col)*${H*15}u+${H*6}u+h*3u;v+=p*vec3<f32>(a1[k],a1[k+1u],a1[k+2u]);}
 var s=v.x*a2[r*13u+d]+v.y*a2[r*13u+3u+d]+v.z*a2[r*13u+6u+d];${zero?'if(a2[r*13u+12u]==0.0){s=0.0;}':''}y[i]=s;`,['f32','f32','f32']);}
export function knnSource(L,K){return kernel(L,`
 var ds:array<f32,${L}>;var ids:array<u32,${L}>;
 for(var j=0u;j<${L}u;j++){
  var dist=1e6+f32(abs(i32(i)-i32(j)))*100.0;
  if(a1[i*13u+12u]>0.0&&a1[j*13u+12u]>0.0){let d=xyz(i*9u+3u)-xyz(j*9u+3u);dist=length(d);}
  if(a2[i]!=a2[j]||a3[i]==0.0||a3[j]==0.0){dist=3.402823e38;}ds[j]=dist;ids[j]=j;
 }
 for(var j=1u;j<${L}u;j++){let v=ds[j];let idx=ids[j];var k=j;loop{if(k==0u){break;}if(ds[k-1u]<=v){break;}ds[k]=ds[k-1u];ids[k]=ids[k-1u];k--;}ds[k]=v;ids[k]=idx;}
 for(var k=0u;k<${K}u;k++){y[i*${K}u+k]=f32(ids[k]);}
 `,['f32','f32','f32','f32'],`fn xyz(k:u32)->vec3<f32>{return vec3<f32>(a0[k],a0[k+1u],a0[k+2u]);}`);}
export function backboneSource(L){return kernel(L,`
 let o=i*23u;let trans=vec3<f32>(a0[o],a0[o+1u],a0[o+2u])*10.0;
 var x=vec3<f32>(a0[o+3u],a0[o+4u],a0[o+5u]);var v=vec3<f32>(a0[o+6u],a0[o+7u],a0[o+8u]);x/=length(x)+1e-5;v/=length(v)+1e-5;
 let nx=x+trans;let ny=v+trans;let rot=gs(trans-nx,ny-trans);
 let n=rot*vec3<f32>(0.5256,1.3612,0.0)+trans;let c=rot*vec3<f32>(-1.5251,0.0,0.0)+trans;
 for(var d=0u;d<3u;d++){y[i*9u+d]=n[d];y[i*9u+3u+d]=trans[d];y[i*9u+6u+d]=c[d];}
 `,['f32'],`fn gs(x:vec3<f32>,v:vec3<f32>)->mat3x3<f32>{let e0=x/sqrt(dot(x,x)+1e-10);let z=v-e0*dot(e0,v);let e1=z/sqrt(dot(z,z)+1e-10);return mat3x3<f32>(e0,e1,cross(e0,e1));}`);}
