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
 var<workgroup> sharedAtoms:array<vec4f,256>;
 @compute @workgroup_size(256) fn main(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32) {
   let wg=group.x+group.y*65535u;
   let c=wg/432u;
   let live=c<28u;
   let sc=min(c,27u);
   let pos=(wg%432u)*256u+lane;
   let origin=vec3f(bitcast<f32>(p[0]),bitcast<f32>(p[1]),bitcast<f32>(p[2]));
   let xyz=origin+0.5*vec3f(vec3u(pos/2304u,(pos/48u)%48u,pos%48u));
   let jStart=starts[sc];
   let jEnd=starts[sc+1u];
   var sum=0.0;
   for(var j0=jStart;j0<jEnd;j0+=256u) {
     let count=min(256u,jEnd-j0);
     if(lane<count) {sharedAtoms[lane]=atoms[j0+lane];}
     workgroupBarrier();
     if(live) {
       for(var k=0u;k<count;k++) {
         let atom=sharedAtoms[k];
         let diff=xyz-atom.xyz;
         let d2=dot(diff,diff)/(atom.w*atom.w);
         if(d2<=1.0){sum+=exp(-2.0*d2);}
         else if(d2<2.25){let t=sqrt(d2)-1.5;sum+=0.54134113295*t*t;}
       }
     }
     workgroupBarrier();
   }
   if(live) {out[c*110592u+pos]=sum;}
 }`,
 conv:`${input} ${weights} ${output} ${params} ${weightRead}
 var<workgroup> features:array<f32,2112>;
 var<workgroup> kernel:array<f32,272>;
 @compute @workgroup_size(256) fn main(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) tid:u32) {
   let d=p[0];let v=d*d*d;let cin=p[1];let cout=p[2];let k=p[3];let k3=k*k*k;let total=cin*k3;
   let left=group.x*128u;let top=group.y*16u;
   let vCol=tid%128u;let kHalf=tid/128u;
   let pos=left+vCol;let posValid=pos<v;
   let coords=vec3i(i32(pos/(d*d)),i32((pos/d)%d),i32(pos%d));
   let pad=i32(k/2u);let dInt=i32(d);
   let kRow=tid/16u;let kCol=tid%16u;
   let laneY=tid/32u;let laneX=tid%32u;
   var a=vec4f(0.0);var b=vec4f(0.0);
   for(var step=0u;step<total;step+=16u) {
     let wRow=top+kRow;let wCol=step+kCol;
     var wVal=0.0;
     if(wRow<cout && wCol<total){wVal=loadw(p[4],p[5],p[6],wRow*total+wCol,wRow);}
     kernel[kRow*17u+kCol]=wVal;
     let kkBase=kHalf*8u;
     if(k==1u) {
       for(var u=0u;u<8u;u++) {
         let kk=kkBase+u;let ch=step+kk;
         var fv=0.0;
         if(posValid && ch<cin){fv=x[ch*v+pos];}
         features[kk*132u+vCol]=fv;
       }
     } else {
       for(var u=0u;u<8u;u++) {
         let kk=kkBase+u;let kidx=step+kk;
         var fv=0.0;
         if(posValid && kidx<total) {
           let ch=kidx/k3;let rem=kidx%k3;
           let delta=vec3i(i32(rem/(k*k)),i32((rem/k)%k),i32(rem%k))-vec3i(pad);
           let at=coords+delta;
           if(all(at>=vec3i(0)) && all(at<vec3i(dInt))) {
             let q=vec3u(at);fv=x[((ch*d+q.x)*d+q.y)*d+q.z];
           }
         }
         features[kk*132u+vCol]=fv;
       }
     }
     workgroupBarrier();
     let fCol=laneX*4u;
     let r0=laneY*2u;let r1=r0+1u;
     for(var j=0u;j<16u;j++) {
       let fb=j*132u+fCol;
       let f=vec4f(features[fb],features[fb+1u],features[fb+2u],features[fb+3u]);
       a=fma(f,vec4f(kernel[r0*17u+j]),a);
       b=fma(f,vec4f(kernel[r1*17u+j]),b);
     }
     workgroupBarrier();
   }
   let outRow=top+laneY*2u;let outCol=left+laneX*4u;
   let slope=bitcast<f32>(p[8]);
   for(var r=0u;r<2u;r++) {
     if(outRow+r<cout) {
       let bias=bitcast<f32>(weights[p[7]+outRow+r]);
       let raw=select(a,b,r==1u)+vec4f(bias);
       let act=select(raw*slope,raw,raw>=vec4f(0.0));
       let baseOut=(outRow+r)*v+outCol;
       for(var j=0u;j<4u;j++) {if(outCol+j<v){out[baseOut+j]=act[j];}}
     }
   }
 }`,
 norm:`${input} ${weights} ${output} ${params}
 @compute @workgroup_size(256) fn main(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32) {
   let i=(group.x+group.y*65535u)*256u+lane;if(i>=p[1]){return;} let ch=i/p[0];
   let scale=bitcast<f32>(weights[p[2]+ch]);let bias=bitcast<f32>(weights[p[3]+ch]);
   let mean=bitcast<f32>(weights[p[4]+ch]);let variance=bitcast<f32>(weights[p[5]+ch]);
   out[i]=(x[i]-mean)*inverseSqrt(variance+1e-5)*scale+bias;
 }`,
 pool:`${input} ${output} ${params}
 @compute @workgroup_size(256) fn main(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32) {
   let i=(group.x+group.y*65535u)*256u+lane;if(i>=p[3]){return;}let k=p[1];let d=p[0];let n=d/k;let ch=i/(n*n*n);
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
 @compute @workgroup_size(256) fn main(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32) {
   let i=(group.x+group.y*65535u)*256u+lane;if(i>=p[1]){return;}
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
