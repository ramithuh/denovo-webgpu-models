const bind = (n, name, type = "f32", write = false) =>
  `@group(0) @binding(${n}) var<storage,${write ? "read_write" : "read"}> ${name}:array<${type}>;`;
const activation = `
fn gelu(x:f32)->f32{
 let a=abs(x)*0.7071067811865476;let t=1.0/(1.0+0.3275911*a);
 let er=1.0-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*exp(-a*a);
 return 0.5*x*(1.0+select(-er,er,x>=0.0));
}`;
export function denseShader(I, O, dtype, gelu, preRelu, useF16) {
  const half = dtype === "F16" && useF16;
  const load =
    dtype === "F32"
      ? "return bitcast<f32>(w[i]);"
      : dtype === "F16"
        ? half
          ? "return f32(w[i]);"
          : "let v=unpack2x16float(w[i/2u]);return v[i%2u];"
        : "let shift=(i%4u)*8u;let v=i32(w[i/4u]<< (24u-shift))>>24;return f32(v)*scale[row];";
  return `${half ? "enable f16;" : ""}
${bind(0, "x")}${bind(1, "w", half ? "f16" : "u32")}${bind(2, "bias")}${bind(3, "scale")}${bind(4, "y", "f32", true)}
fn weight(i:u32,row:u32)->f32{${load}}
${activation}
var<workgroup> at:array<f32,256>;var<workgroup> bt:array<f32,256>;
@compute @workgroup_size(8,8) fn main(@builtin(local_invocation_id) local:vec3<u32>,@builtin(workgroup_id) group:vec3<u32>){
 let row=group.y*8u+local.y;let col=group.x*8u+local.x;let lane=local.y*8u+local.x;
 let rows=arrayLength(&x)/${I}u;var sum=0.0;
 for(var start=0u;start<${I}u;start+=32u){
  for(var t=lane;t<256u;t+=64u){
    let r=group.y*8u+t/32u;let k=start+t%32u;
    var a=0.0;if(r<rows && k<${I}u){a=x[r*${I}u+k];}at[t]=${preRelu ? "max(a,0.0)" : "a"};
    let c=group.x*8u+t/32u;var b=0.0;if(c<${O}u && k<${I}u){b=weight(c*${I}u+k,c);}bt[t]=b;
  }
  workgroupBarrier();
  for(var k=0u;k<32u;k++){sum+=at[local.y*32u+k]*bt[local.x*32u+k];}
  workgroupBarrier();
 }
 if(row<rows && col<${O}u){let value=sum+bias[col];y[row*${O}u+col]=${gelu ? "gelu(value)" : "value"};}
}`;
}
export function normShader(delta, masked) {
  return `
${bind(0, "x")}${bind(1, "d")}${bind(2, "w")}${bind(3, "b")}${bind(4, "mask")}${bind(5, "y", "f32", true)}
var<workgroup> values:array<f32,128>;var<workgroup> sums:array<f32,128>;
@compute @workgroup_size(128) fn main(@builtin(local_invocation_id) lid:vec3<u32>,@builtin(workgroup_id) wid:vec3<u32>){
 let c=lid.x;let i=wid.x*128u+c;let v=x[i]${delta ? "+d[i]" : ""};values[c]=v;sums[c]=v;workgroupBarrier();
 for(var step=64u;step>0u;step/=2u){if(c<step){sums[c]+=sums[c+step];}workgroupBarrier();}
 let mean=sums[0]/128.0;workgroupBarrier();sums[c]=(v-mean)*(v-mean);workgroupBarrier();
 for(var step=64u;step>0u;step/=2u){if(c<step){sums[c]+=sums[c+step];}workgroupBarrier();}
 let inv=inverseSqrt(sums[0]/128.0+0.00001);y[i]=((v-mean)*inv*w[c]+b[c])${masked ? "*mask[wid.x]" : ""};
}`;
}
export function xyzShader() {
  return `${bind(0, "x")}${bind(1, "y", "f32", true)}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i>=arrayLength(&x)/12u){return;}
 for(var c=0u;c<12u;c++){y[i*15u+c]=x[i*12u+c];}
 let n=vec3<f32>(x[i*12u],x[i*12u+1u],x[i*12u+2u]);let ca=vec3<f32>(x[i*12u+3u],x[i*12u+4u],x[i*12u+5u]);let cc=vec3<f32>(x[i*12u+6u],x[i*12u+7u],x[i*12u+8u]);
 let b=ca-n;let c=cc-ca;let a=cross(b,c);let cb=((-0.58273431*a+0.56802827*b)-0.54067466*c)+ca;
 y[i*15u+12u]=cb.x;y[i*15u+13u]=cb.y;y[i*15u+14u]=cb.z;
}`;
}
export function distanceShader(L) {
  return `${bind(0, "xyz")}${bind(1, "mask")}${bind(2, "d", "f32", true)}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i>=${L}u){return;}var maximum=0.0;
 for(var j=0u;j<${L}u;j++){var sum=0.0;for(var c=0u;c<3u;c++){let diff=xyz[i*15u+3u+c]-xyz[j*15u+3u+c];sum+=diff*diff;}
 let value=sqrt(sum+0.000001)*mask[i]*mask[j];d[i*${L}u+j]=value;maximum=max(maximum,value);}
 for(var j=0u;j<${L}u;j++){d[i*${L}u+j]+=(1.0-mask[i]*mask[j])*maximum;}
}`;
}
export function topkShader(L, K) {
  return `${bind(0, "d")}${bind(1, "indices", "u32", true)}
var<private> q:array<u32,${L}>;var<private> base:u32;
fn less(a:u32,b:u32)->bool{return d[base+a]<d[base+b];}
fn swap(a:u32,b:u32){let t=q[a];q[a]=q[b];q[b]=t;}
fn median(out:u32,a:u32,b:u32,c:u32){
 if(less(q[a],q[b])){if(less(q[b],q[c])){swap(out,b);}else if(less(q[a],q[c])){swap(out,c);}else{swap(out,a);}}
 else if(less(q[a],q[c])){swap(out,a);}else if(less(q[b],q[c])){swap(out,c);}else{swap(out,b);}
}
fn splitRange(lo:u32,hi:u32)->u32{
 median(lo,lo+1u,lo+(hi-lo)/2u,hi-1u);let pivot=q[lo];var a=lo+1u;var b=hi;
 loop{while(less(q[a],pivot)){a++;}b--;while(less(pivot,q[b])){b--;}if(a>=b){return a;}swap(a,b);a++;}
}
fn insertion(lo:u32,hi:u32){for(var i=lo+1u;i<hi;i++){let value=q[i];var j=i;while(j>lo){if(!less(value,q[j-1u])){break;}q[j]=q[j-1u];j--;}q[j]=value;}}
fn adjust(lo:u32,h:u32,len:u32,value:u32){
 let top=h;var hole=h;var child=hole;
 while(child<(len-1u)/2u){child=2u*(child+1u);if(less(q[lo+child],q[lo+child-1u])){child--;}q[lo+hole]=q[lo+child];hole=child;}
 if(len%2u==0u && child==(len-2u)/2u){child=2u*(child+1u);q[lo+hole]=q[lo+child-1u];hole=child-1u;}
 var parent=(hole-1u)/2u;
 while(hole>top){if(!less(q[lo+parent],value)){break;}q[lo+hole]=q[lo+parent];hole=parent;parent=(hole-1u)/2u;}
 q[lo+hole]=value;
}
fn heap(lo:u32,hi:u32){let n=hi-lo;if(n<2u){return;}var p=i32((n-2u)/2u);while(p>=0){adjust(lo,u32(p),n,q[lo+u32(p)]);p--;}}
fn heapsort(lo:u32,hi:u32){heap(lo,hi);var end=hi;while(end-lo>1u){let v=q[end-1u];q[end-1u]=q[lo];end--;adjust(lo,0u,end-lo,v);}}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3<u32>){
 let row=id.x;base=row*${L}u;for(var i=0u;i<${L}u;i++){q[i]=i;}
 var lo=0u;var hi=${L}u;var depth=${2 * Math.floor(Math.log2(L))}u;
 while(hi-lo>3u){
  if(depth==0u){heap(lo,${K}u);for(var i=${K}u;i<hi;i++){if(less(q[i],q[lo])){let v=q[i];q[i]=q[lo];adjust(lo,0u,${K}u-lo,v);}}swap(lo,${K - 1}u);break;}
  depth--;let cut=splitRange(lo,hi);if(cut<=${K - 1}u){lo=cut;}else{hi=cut;}
 }
 if(hi-lo<=3u){insertion(lo,hi);}
 var stack:array<vec3<u32>,32>;var size=1u;stack[0]=vec3<u32>(0u,${K - 1}u,${2 * Math.floor(Math.log2(Math.max(1, K - 1)))}u);
 while(size>0u){size--;let s=stack[size];lo=s.x;hi=s.y;depth=s.z;
  while(hi-lo>16u){if(depth==0u){heapsort(lo,hi);break;}depth--;let cut=splitRange(lo,hi);stack[size]=vec3<u32>(cut,hi,depth);size++;hi=cut;}
 }
 insertion(0u,${K - 1}u);for(var k=0u;k<${K}u;k++){indices[row*${K}u+k]=q[k];}
}`;
}
export function geometryShader(L, K) {
  return `${bind(0, "xyz")}${bind(1, "dist")}${bind(2, "idx", "u32")}${bind(3, "residue", "i32")}${bind(4, "chain", "i32")}${bind(5, "w")}${bind(6, "b")}${bind(7, "out", "f32", true)}
const aa=array<u32,25>(1,0,2,3,4,1,1,1,1,0,0,0,4,4,3,0,2,3,4,2,3,4,2,3,2);
const bb=array<u32,25>(1,0,2,3,4,0,2,3,4,2,3,4,2,3,2,1,1,1,1,0,0,0,4,4,3);
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){
 let x=id.x;if(x>=${L * K * 416}u){return;}let r=x/416u;let c=x%416u;let i=r/${K}u;let j=idx[r];
 if(c<16u){let offset=select(65u,u32(clamp(residue[i]-residue[j]+32,0,64)),chain[i]==chain[j]);out[x]=w[c*66u+offset]+b[c];return;}
 let pair=(c-16u)/16u;let channel=(c-16u)%16u;var distance=dist[i*${L}u+j];
 if(pair>0u){var sum=0.0;for(var k=0u;k<3u;k++){let delta=xyz[i*15u+aa[pair]*3u+k]-xyz[j*15u+bb[pair]*3u+k];sum+=delta*delta;}distance=sqrt(sum+0.000001);}
 let step=20.0/15.0;let mu=select(22.0-f32(15u-channel)*step,2.0+f32(channel)*step,channel<8u);let z=(distance-mu)/1.25;out[x]=exp(-z*z);
}`;
}
export function concatShader(L, K, decoder) {
  const C = decoder ? 512 : 384;
  return `${bind(0, "v")}${bind(1, "e")}${bind(2, "idx", "u32")}${bind(3, "s")}${bind(4, "mask")}${bind(5, "out", "f32", true)}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){let x=id.x;if(x>=${L * K * C}u){return;}let r=x/${C}u;let c=x%${C}u;let i=r/${K}u;let j=idx[r];
 if(c<128u){out[x]=v[i*128u+c];}else if(c<256u){out[x]=e[r*128u+c-128u]${decoder ? "*mask[i]" : ""};}
 ${decoder ? "else if(c<384u){out[x]=s[j*128u+c-256u]*mask[i];}else{out[x]=v[j*128u+c-384u]*mask[i];}" : "else{out[x]=v[j*128u+c-256u];}"}
}`;
}
export function reduceShader(L, K, encoder) {
  return `${bind(0, "x")}${bind(1, "idx", "u32")}${bind(2, "mask")}${bind(3, "out", "f32", true)}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){let x0=id.x;if(x0>=${L * 128}u){return;}let i=x0/128u;let c=x0%128u;var sum=0.0;
 for(var k=0u;k<${K}u;k++){sum+=x[(i*${K}u+k)*128u+c]${encoder ? "*mask[i]*mask[idx[i*" + K + "u+k]]" : ""};}out[x0]=sum/30.0;
}`;
}
export function embedShader() {
  return `${bind(0, "s", "u32")}${bind(1, "w")}${bind(2, "out", "f32", true)}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i>=arrayLength(&out)){return;}out[i]=w[s[i/128u]*128u+i%128u];}`;
}
export function joinShader() {
  return `${bind(0, "a")}${bind(1, "b")}${bind(2, "c")}${bind(3, "out", "f32", true)}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i>=arrayLength(&out)){return;}let r=i/384u;let ch=i%384u;
 if(ch<128u){out[i]=a[r*128u+ch];}else if(ch<256u){out[i]=b[r*128u+ch-128u];}else{out[i]=c[r*128u+ch-256u];}}`;
}
export function calibrateShader() {
  return `${bind(0, "x")}${bind(1, "w")}${bind(2, "b")}${bind(3, "out", "f32", true)}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i>=arrayLength(&out)){return;}let product=x[i]*w[0];out[i]=product+b[0];}`;
}
export function subtractShader() {
  return `${bind(0, "scores")}${bind(1, "s", "u32")}${bind(2, "out", "f32", true)}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i>=arrayLength(&out)){return;}let r=i/20u;out[i]=scores[r*21u+i%20u]-scores[r*21u+s[r]];}`;
}
