const params = `struct Params { L:u32, CI:u32, CO:u32, KW:u32, D:u32, act:u32, mode:u32, pad:u32 };`;
export const embedding = `${params}
@group(0) @binding(0) var<storage,read> tokens:array<u32>;
@group(0) @binding(1) var<storage,read> table:array<f32>;
@group(0) @binding(2) var<storage,read_write> y:array<f32>;
@group(0) @binding(3) var<uniform> p:Params;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
 let i=gid.x; if(i<p.L*8u) {y[i]=table[tokens[i/8u]*8u+i%8u];}
}`;
const gelu = `fn gelu(x:f32)->f32 {
 let z=abs(x)*0.7071067811865475; let t=1.0/(1.0+0.3275911*z);
 let e=1.0-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*exp(-z*z);
 return x*0.5*(1.0+select(e,-e,x<0.0));
}`;
export const norm = `${params}${gelu}
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> w:array<f32>;
@group(0) @binding(2) var<storage,read> b:array<f32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> reduction:array<f32,256>;
var<workgroup> mean:f32;
var<workgroup> inv:f32;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) t:u32,@builtin(workgroup_id) g:vec3<u32>) {
 let base=g.x*p.CI; var sum=0.0;
 for(var c=t;c<p.CI;c+=256u){sum+=x[base+c];} reduction[t]=sum; workgroupBarrier();
 for(var stride=128u;stride>0u;stride/=2u){if(t<stride){reduction[t]+=reduction[t+stride];} workgroupBarrier();}
 if(t==0u){mean=reduction[0]/f32(p.CI);} workgroupBarrier();
 sum=0.0;for(var c=t;c<p.CI;c+=256u){let d=x[base+c]-mean;sum+=d*d;} reduction[t]=sum;workgroupBarrier();
 for(var stride=128u;stride>0u;stride/=2u){if(t<stride){reduction[t]+=reduction[t+stride];}workgroupBarrier();}
 if(t==0u){inv=inverseSqrt(reduction[0]/f32(p.CI)+0.00001);}workgroupBarrier();
 for(var c=t;c<p.CI;c+=256u){let v=(x[base+c]-mean)*inv*w[c]+b[c];y[base+c]=select(v,gelu(v),p.act==1u);}
}`;
export const residual = `${params}
@group(0) @binding(0) var<storage,read> delta:array<f32>;
@group(0) @binding(1) var<storage,read_write> x:array<f32>;
@group(0) @binding(2) var<uniform> p:Params;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>){if(g.x<p.L*1024u){x[g.x]+=delta[g.x];}}
`;
export function convolution(dtype, useF16 = false) {
  const nativeHalf = dtype === "F16" && useF16;
  const fetch =
    dtype === "F32"
      ? "return bitcast<f32>(w[i]);"
      : dtype === "F16"
        ? nativeHalf
          ? "return f32(w[i]);"
          : "let pair=unpack2x16float(w[i/2u]);return select(pair.x,pair.y,(i%2u)==1u);"
        : "let v=i32((w[i/4u]>>((i%4u)*8u))&255u);return f32(select(v,v-256,v>=128))*scale[o];";
  return `${nativeHalf ? "enable f16;" : ""}${params}
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> w:array<${nativeHalf ? "f16" : "u32"}>;
@group(0) @binding(2) var<storage,read> bias:array<f32>;
@group(0) @binding(3) var<storage,read> scale:array<f32>;
@group(0) @binding(4) var<storage,read> mask:array<f32>;
@group(0) @binding(5) var<storage,read_write> y:array<f32>;
@group(0) @binding(6) var<uniform> p:Params;
fn weight(i:u32,o:u32)->f32{${fetch}}
var<workgroup> tileA:array<f32,256>;
var<workgroup> tileB:array<f32,1024>;
@compute @workgroup_size(8,8) fn main(@builtin(local_invocation_id) t:vec3<u32>,@builtin(workgroup_id) g:vec3<u32>) {
 let tid=t.y*8u+t.x;let row=g.y*8u+t.y;let col=g.x*32u+t.x*4u;
 var acc=vec4<f32>(0.0);let K=p.CI*p.KW;
 for(var start=0u;start<K;start+=32u){
   for(var j=tid;j<256u;j+=64u){
     let r=g.y*8u+j/32u;let k=start+j%32u;let pos=i32(r)+(i32(k%p.KW)-i32(p.KW/2u))*i32(p.D);
     var v=0.0;
     if(r<p.L&&k<K&&pos>=0&&pos<i32(p.L)){v=x[u32(pos)*p.CI+k/p.KW];if(p.mode==1u){v*=mask[u32(pos)];}}
     tileA[j]=v;
   }
   for(var j=tid;j<1024u;j+=64u){let k=start+j/32u;let o=g.x*32u+j%32u;var v=0.0;if(k<K&&o<p.CO){v=weight(k*p.CO+o,o);}tileB[j]=v;}
   workgroupBarrier();
   for(var k=0u;k<32u;k++){
     let a=tileA[t.y*32u+k];let idx=k*32u+t.x*4u;
     acc+=a*vec4<f32>(tileB[idx],tileB[idx+1u],tileB[idx+2u],tileB[idx+3u]);
   }
   workgroupBarrier();
 }
 if(row<p.L){for(var j=0u;j<4u;j++){if(col+j<p.CO){y[row*p.CO+col+j]=acc[j]+bias[col+j];}}}
}`;
}
