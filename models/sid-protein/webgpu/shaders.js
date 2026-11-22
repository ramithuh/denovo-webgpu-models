import { numericWGSL, divisionWGSL } from "../shared/numeric.js";
/** Shape parameters are storage data; pipelines are independent of L and T. */
export const decl = (i, n, type = "f32", write = false) =>
  `@group(0) @binding(${i}) var<storage,${write ? "read_write" : "read"}> ${n}:array<${type}>;\n`;
export function weightReader(dtype, native = false) {
  return dtype === "F32"
    ? "return w[i];"
    : dtype === "F16"
      ? native
        ? "return f32(w[i]);"
        : "return unpack2x16float(w[i/2u])[i%2u];"
      : "let q=i32((w[i/4u]>>((i%4u)*8u))&255u);return f32(select(q,q-256,q>=128))*sc[row];";
}
export function matmul(dtype, hasBias, native = false) {
  return `${native ? "enable f16;" : ""}
${decl(0, "a")}${decl(1, "w", dtype === "F32" ? "f32" : native ? "f16" : "u32")}${decl(2, "b")}${decl(3, "sc")}${decl(4, "p", "u32")}${decl(5, "y", "f32", true)}
fn weight(i:u32,row:u32)->f32{${weightReader(dtype, native)}}
var<workgroup> at:array<f32,256>;var<workgroup> bt:array<f32,256>;
@compute @workgroup_size(16,16) fn main(@builtin(local_invocation_id) tid:vec3<u32>,@builtin(workgroup_id) gid:vec3<u32>){
 let M=p[0];let N=p[1];let K=p[2];let r=gid.y*16u+tid.y;let c=gid.x*16u+tid.x;let idx=tid.y*16u+tid.x;var s=0.0;${hasBias ? "if(c<N){s=b[c];}" : ""}
 for(var base=0u;base<K;base+=16u){var av=0.0;var bv=0.0;if(r<M&&base+tid.x<K){av=a[r*K+base+tid.x];}if(c<N&&base+tid.y<K){bv=weight(c*K+base+tid.y,c);}at[idx]=av;bt[idx]=bv;workgroupBarrier();
 for(var k=0u;k<16u;k++){s=fma(at[tid.y*16u+k],bt[k*16u+tid.x],s);}workgroupBarrier();}
 if(r<M&&c<N){y[r*N+c]=s;}
}`;
}
// Cascaded four-lane Welford moments follow the pinned ARM PyTorch oracle.
export const norm = `${divisionWGSL}
${decl(0, "a")}${decl(1, "w")}${decl(2, "b")}${decl(3, "p", "u32")}${decl(4, "y", "f32", true)}
var<workgroup> means:array<f32,4>;var<workgroup> variances:array<f32,4>;
var<workgroup> mu:f32;var<workgroup> inv:f32;
fn divide(x:f32,z:f32)->f32{if(z==0.0){return 0.0;}return rfdiv(x,z);}
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) gid:vec3<u32>){
let rawRow=gid.x+gid.y*65535u;let row=min(rawRow,p[4]-1u);let D=p[0];let stride=p[1];let offset=p[2];
if(lane<4u){
 var count:array<u32,8>;var ms:array<f32,8>;var vs:array<f32,8>;
 let n=D/4u;let chunks=(n+15u)/16u;var depth=0u;var z=chunks-1u;while(z>0u){depth++;z=z>>1u;}
 for(var chunk=0u;chunk<chunks;chunk++){
  let size=min(16u,n-chunk*16u);var m=0.0;var v=0.0;
  for(var j=0u;j<size;j++){let x=a[row*stride+offset+(chunk*16u+j)*4u+lane];let delta=x-m;m=fma(delta,divide(1.0,f32(j+1u)),m);v=fma(delta,x-m,v);}
  var total=count[0]+size;var c=divide(f32(size),f32(total));var delta=m-ms[0];ms[0]=fma(c,delta,ms[0]);vs[0]=vs[0]+fma((delta*delta)*c,f32(count[0]),v);count[0]=total;
  var mask=chunk+1u;var level=1u;
  while(level<depth&&(mask&1u)==0u){
   total=count[level]+count[level-1u];c=divide(f32(count[level-1u]),f32(total));delta=ms[level-1u]-ms[level];ms[level]=fma(c,delta,ms[level]);vs[level]=vs[level]+fma((delta*delta)*c,f32(count[level]),vs[level-1u]);count[level]=total;
   count[level-1u]=0u;ms[level-1u]=0.0;vs[level-1u]=0.0;mask=mask>>1u;level++;
  }
 }
 for(var level=1u;level<depth;level++){
  let total=count[0]+count[level];let c=divide(f32(count[level]),f32(total));let delta=ms[level]-ms[0];ms[0]=fma(c,delta,ms[0]);vs[0]=vs[0]+fma((delta*delta)*c,f32(count[0]),vs[level]);count[0]=total;
 }
 means[lane]=ms[0];variances[lane]=vs[0];
}
workgroupBarrier();
if(lane==0u){var count=0u;var m=0.0;var v=0.0;for(var k=0u;k<4u;k++){let next=count+D/4u;let c=divide(f32(D/4u),f32(next));let delta=means[k]-m;m=fma(c,delta,m);v=v+fma((delta*delta)*c,f32(count),variances[k]);count=next;}mu=m;let variance=divide(v,f32(D))+0.00001;var root=sqrt(variance);root=root+fma(-root,root,variance)/(2.0*root);inv=divide(1.0,root);}
workgroupBarrier();
for(var j=lane;j<D;j+=256u){var z=(a[row*stride+offset+j]-mu)*inv;if(p[3]!=0u){z=fma(z,w[j],b[j]);}if(rawRow<p[4]){y[row*D+j]=z;}}
}`;
export function element(body) {
  return `${decl(0, "a")}${decl(1, "b")}${decl(2, "c")}${decl(3, "p", "u32")}${decl(4, "y", "f32", true)}
${numericWGSL}${divisionWGSL}
fn sigmoid(x:f32)->f32{return rfdiv(1.0,1.0+rfExp(-x));}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>){let i=g.x+g.y*16776960u;if(i>=p[0]){return;}let D=p[1];${body}}`;
}
export const elements = {
  add: element("y[i]=a[i]+b[i];"),
  swiglu: element(
    "let j=(i/D)*D*2u+i%D;let v=a[j+D];y[i]=rfdiv(v,1.0+rfExp(-v))*a[j];",
  ),
  adaptiveProduct: element(
    "let row=select(0u,select(0u,1u,i/D>=10u),p[2]!=0u);let j=row*D+i%D;y[i]=a[i]*sigmoid(b[j]);",
  ),
  adaptiveBias: element(
    "let row=select(0u,select(0u,1u,i/D>=10u),p[2]!=0u);let j=row*D+i%D;y[i]=a[i]+b[j];",
  ),
  residualProduct: element(
    "let j=select(0u,D,i/D>=10u)+i%D;y[i]=a[i]*sigmoid(b[j]);",
  ),
  padCondition: element("y[i]=0.0;if(i>=512u){y[i]=a[i-512u];}"),
  padPair: element(
    "let N=p[2];let row=(i/D)/N;let col=(i/D)%N;y[i]=0.0;if(row>=10u&&col>=10u){y[i]=a[((row-10u)*(N-10u)+col-10u)*D+i%D];}",
  ),
  unpad: element("y[i]=a[i+p[2]*D];"),
  velocityProduct: element("let t=bitcast<f32>(p[2]);y[i]=(1.0-t)*a[30u+i];"),
  time: element(
    "let k=i%128u;let v=bitcast<f32>(p[2])*2000.0*a[k];y[i]=select(rfSin(v),rfCos(v),i>=128u);",
  ),
  position: element(
    // ARM SLEEF chooses argument reduction for a whole four-lane vector.
    // Frequencies decrease within each block, so its first lane is largest.
    "let row=i/132u;let k=i%132u;y[i]=0.0;if(k<128u){let numerator=f32(row+1u)*3.1415927410125732;let z=rfdiv(numerator,a[k%64u]);let large=rfdiv(numerator,a[((k%64u)/4u)*4u])>=125.0;let sc=rfSinCos(z,large);y[i]=select(sc.x,sc.y,k>=64u);}",
  ),
};
export function gather(dtype, native = false) {
  return `${native ? "enable f16;" : ""}${decl(0, "a")}${decl(1, "w", dtype === "F32" ? "f32" : native ? "f16" : "u32")}${decl(2, "sc")}${decl(3, "p", "u32")}${decl(4, "y", "f32", true)}
fn weight(i:u32,row:u32)->f32{${weightReader(dtype, native)}}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>){let i=g.x;if(i>=p[0]){return;}if(p[1]==0u){y[i]=weight(p[2]*256u+i,p[2]);}else{if(i<7680u){y[i]=weight(i,i/768u);}else{y[i]=a[i-7680u];}}}`;
}
export const pairFeatures = `${decl(0, "x")}${decl(1, "bounds")}${decl(2, "p", "u32")}${decl(3, "y", "f32", true)}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>){let i=g.x+g.y*16776960u;let L=p[0];if(i>=L*L*319u){return;}let row=(i/319u)/L;let col=(i/319u)%L;let d=i%319u;let sep=u32(clamp(i32(row)-i32(col)+63,0,126));
var bucket=0u;let dx=x[row*3u]-x[col*3u];let dy=x[row*3u+1u]-x[col*3u+1u];let dz=x[row*3u+2u]-x[col*3u+2u];let dist=sqrt((dx*dx+dy*dy)+dz*dz);for(var j=0u;j<63u;j++){if(dist>bounds[j]){bucket++;}}
y[i]=select(0.0,1.0,d==sep||d==255u+bucket);}`;
export const scores = `${decl(0, "q")}${decl(1, "k")}${decl(2, "b")}${decl(3, "p", "u32")}${decl(4, "y", "f32", true)}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){let i=g.x+g.y*4194240u;let N=p[0];if(i>=12u*N*N){return;}let h=i/(N*N);let row=(i/N)%N;let col=i%N;var s=0.0;for(var d=0u;d<64u;d++){s+=q[row*768u+h*64u+d]*k[col*768u+h*64u+d];}y[i]=s*0.125+b[(row*N+col)*12u+h];}`;
export const softmax = `${numericWGSL}${divisionWGSL}${decl(0, "x")}${decl(1, "p", "u32")}${decl(2, "y", "f32", true)}
var<workgroup> tmp:array<f32,256>;var<workgroup> inverse:f32;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) g:vec3<u32>){
let N=p[0];let off=g.x*N;var m=-3.402823e38;for(var k=lane;k<N;k+=256u){m=max(m,x[off+k]);}tmp[lane]=m;workgroupBarrier();
for(var n=128u;n>0u;n/=2u){if(lane<n){tmp[lane]=max(tmp[lane],tmp[lane+n]);}workgroupBarrier();}m=tmp[0];workgroupBarrier();
if(lane<4u){var s=0.0;for(var k=lane;k<N;k+=4u){s+=rfExp(x[off+k]-m);}tmp[lane]=s;}workgroupBarrier();
if(lane==0u){let den=(tmp[0]+tmp[2])+(tmp[1]+tmp[3]);inverse=rfdiv(1.0,den);}workgroupBarrier();
for(var k=lane;k<N;k+=256u){y[off+k]=rfExp(x[off+k]-m)*inverse;}}`;
export const context = `${numericWGSL}${divisionWGSL}
${decl(0, "a")}${decl(1, "qkv")}${decl(2, "gate")}${decl(3, "p", "u32")}${decl(4, "y", "f32", true)}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>){let i=g.x;let N=p[0];if(i>=N*768u){return;}let row=i/768u;let ch=i%768u;let h=ch/64u;var s=0.0;for(var j=0u;j<N;j++){s+=a[(h*N+row)*N+j]*qkv[j*2304u+1536u+ch];}y[i]=s*rfdiv(1.0,1.0+rfExp(-gate[i]));}`;
export const coordinateMean = `${divisionWGSL}${decl(0, "x")}${decl(1, "p", "u32")}${decl(2, "y", "f32", true)}
@compute @workgroup_size(4) fn main(@builtin(global_invocation_id) g:vec3<u32>){let d=g.x;if(d>=3u){return;}let L=p[0];let size=L/4u;var levelPower=4u;var lg=0u;var q=max(size,1u)-1u;while(q>0u){lg++;q=q>>1u;}levelPower=max(levelPower,lg/4u);let step=1u<<levelPower;let mask=step-1u;var acc:array<f32,16>;var i=0u;
while(i+step<=size){for(var j=0u;j<step;j++){for(var k=0u;k<4u;k++){acc[k]+=x[((i+j)*4u+k)*3u+d];}}i+=step;for(var level=1u;level<4u;level++){for(var k=0u;k<4u;k++){acc[level*4u+k]+=acc[(level-1u)*4u+k];acc[(level-1u)*4u+k]=0.0;}if((i&(mask<<(level*levelPower)))!=0u){break;}}}
while(i<size){for(var k=0u;k<4u;k++){acc[k]+=x[(i*4u+k)*3u+d];}i++;}
for(var level=1u;level<4u;level++){for(var k=0u;k<4u;k++){acc[k]+=acc[level*4u+k];}}
for(var j=size*4u;j<L;j++){acc[0]+=x[j*3u+d];}var sum=acc[0];for(var k=1u;k<4u;k++){sum+=acc[k];}y[d]=rfdiv(sum,f32(L));}`;
elements.center = element("y[i]=a[i]-b[i%3u];");
elements.scale = element("y[i]=a[i]*bitcast<f32>(p[2]);");
