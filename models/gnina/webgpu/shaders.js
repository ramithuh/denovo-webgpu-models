// Buffers use NCDHW order. All weight offsets count 32-bit words.
const input="@group(0) @binding(0) var<storage,read> x:array<f32>;";
const weights="@group(0) @binding(1) var<storage,read> weights:array<u32>;";
const output="@group(0) @binding(2) var<storage,read_write> out:array<f32>;";
const params="@group(0) @binding(3) var<storage,read> p:array<u32>;";
const weightRead=`
fn loadw(base:u32,scale:u32,kind:u32,i:u32,row:u32)->f32 {
  if(kind==0u) {return bitcast<f32>(weights[base+i]);}
  if(kind==1u) {let pair=unpack2x16float(weights[base+i/2u]);return select(pair.x,pair.y,(i&1u)==1u);}
  let packed=weights[base+i/4u]; let shift=(i&3u)*8u;
  let signedByte=bitcast<i32>(((packed>>shift)&255u)<<24u)>>24u;
  return f32(signedByte)*bitcast<f32>(weights[scale+row]);
}`;
export const shaders={
 grid:`@group(0) @binding(0) var<storage,read> atoms:array<vec4f>;
 @group(0) @binding(1) var<storage,read> starts:array<u32>;
 ${output} ${params}
 @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) gid:vec3u) {
   let i=gid.x; let c=i/110592u; if(c>=28u){return;}
   let pos=i%110592u;
   let origin=vec3f(bitcast<f32>(p[0]),bitcast<f32>(p[1]),bitcast<f32>(p[2]));
   let xyz=origin+0.5*vec3f(vec3u(pos/2304u,(pos/48u)%48u,pos%48u));
   var sum=0.0;
   for(var j=starts[c];j<starts[c+1u];j++) {
     let atom=atoms[j];let diff=xyz-atom.xyz;let d2=dot(diff,diff)/(atom.w*atom.w);
     if(d2<=1.0){sum+=exp(-2.0*d2);}
     else if(d2<2.25){let t=sqrt(d2)-1.5;sum+=0.54134113295*t*t;}
   }
   out[i]=sum;
 }`,
 conv:`${input} ${weights} ${output} ${params} ${weightRead}
 var<workgroup> features:array<array<f32,32>,8>;
 var<workgroup> kernel:array<array<f32,8>,16>;
 fn feature(pos:u32,kidx:u32,d:u32,k:u32,cin:u32)->f32 {
   let v=d*d*d;let ch=kidx/(k*k*k); if(pos>=v || ch>=cin){return 0.0;}
   if(k==1u){return x[ch*v+pos];}
   let coords=vec3i(i32(pos/(d*d)),i32(pos/d%d),i32(pos%d));
   let delta=vec3i(i32(kidx/(k*k)%k),i32(kidx/k%k),i32(kidx%k))-vec3i(i32(k/2u));
   let at=coords+delta;
   if(any(at<vec3i(0))||any(at>=vec3i(i32(d)))){return 0.0;}
   let q=vec3u(at);return x[((ch*d+q.x)*d+q.y)*d+q.z];
 }
 @compute @workgroup_size(8,8) fn main(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_id) lane:vec3u) {
   let d=p[0];let v=d*d*d;let cin=p[1];let cout=p[2];let k=p[3];let total=cin*k*k*k;
   let left=group.x*32u;let top=group.y*16u;let tid=lane.y*8u+lane.x;
   var a=vec4f(0.0);var b=vec4f(0.0);
   for(var step=0u;step<total;step+=8u) {
     for(var n=tid;n<256u;n+=64u){features[n/32u][n%32u]=feature(left+n%32u,step+n/32u,d,k,cin);}
     for(var n=tid;n<128u;n+=64u){
       let row=top+n/8u;let col=step+n%8u;
       var val=0.0;if(row<cout&&col<total){val=loadw(p[4],p[5],p[6],row*total+col,row);}
       kernel[n/8u][n%8u]=val;
     }
     workgroupBarrier();
     for(var j=0u;j<8u;j++){
       let col=lane.x*4u;
       let f=vec4f(features[j][col],features[j][col+1u],features[j][col+2u],features[j][col+3u]);
       a=fma(f,vec4f(kernel[lane.y*2u][j]),a);
       b=fma(f,vec4f(kernel[lane.y*2u+1u][j]),b);
     }
     workgroupBarrier();
   }
   let row=top+lane.y*2u;let col=left+lane.x*4u;
   for(var r=0u;r<2u;r++){
     if(row+r<cout){let bias=bitcast<f32>(weights[p[7]+row+r]);let raw=select(a,b,r==1u)+vec4f(bias);let act=select(raw*bitcast<f32>(p[8]),raw,raw>=vec4f(0));
       for(var j=0u;j<4u;j++){if(col+j<v){out[(row+r)*v+col+j]=act[j];}}
     }
   }
 }`,
 norm:`${input} ${weights} ${output} ${params}
 @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) gid:vec3u) {
   let i=gid.x;if(i>=p[1]){return;} let ch=i/p[0];
   let scale=bitcast<f32>(weights[p[2]+ch]);let bias=bitcast<f32>(weights[p[3]+ch]);
   let mean=bitcast<f32>(weights[p[4]+ch]);let variance=bitcast<f32>(weights[p[5]+ch]);
   out[i]=(x[i]-mean)*inverseSqrt(variance+1e-5)*scale+bias;
 }`,
 pool:`${input} ${output} ${params}
 @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) gid:vec3u) {
   let i=gid.x;if(i>=p[3]){return;}let k=p[1];let d=p[0];let n=d/k;let ch=i/(n*n*n);
   let ox=i/(n*n)%n*k;let oy=i/n%n*k;let oz=i%n*k;
   var val=select(-3.402823e38,0.0,p[2]==1u);
   for(var a=0u;a<k;a++){for(var b=0u;b<k;b++){for(var e=0u;e<k;e++){
     let f=x[((ch*d+ox+a)*d+oy+b)*d+oz+e];if(p[2]==1u){val+=f;}else{val=max(val,f);}
   }}}
   out[i]=select(val,val/f32(k*k*k),p[2]==1u);
 }`,
 concat:`${input}
 @group(0) @binding(1) var<storage,read> y:array<f32>;
 ${output} ${params}
 @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) gid:vec3u) {
   let i=gid.x;if(i>=p[1]){return;}
   if(i<p[0]){out[i]=x[i];}else{out[i]=y[i-p[0]];}
 }`,
 linear:`${input} ${weights} ${output} ${params} ${weightRead}
 var<workgroup> partial:array<f32,128>;
 @compute @workgroup_size(128) fn main(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
   let row=group.x;let n=p[0];let base=select(1u,5u,row==2u);let r=select(row,0u,row==2u);
   var sum=0.0;for(var j=lane;j<n;j+=128u){sum=fma(x[j],loadw(p[base],p[base+1u],p[base+2u],r*n+j,r),sum);}
   partial[lane]=sum;workgroupBarrier();
   for(var step=64u;step>0u;step/=2u){if(lane<step){partial[lane]+=partial[lane+step];}workgroupBarrier();}
   if(lane==0u){out[row]=partial[0]+bitcast<f32>(weights[p[base+3u]+r]);}
 }`,
 softmax:`${input} ${output}
 @compute @workgroup_size(1) fn main(){out[0]=1.0/(1.0+exp(x[0]-x[1]));out[1]=x[2];}`,
};
