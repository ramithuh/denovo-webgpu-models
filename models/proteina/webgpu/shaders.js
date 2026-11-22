// All dimensions arrive through uniforms. No shape/step-specialized shader source.
export const params = `struct P { n:u32,c:u32,k:u32,m:u32,a:u32,b:u32,d:u32,e:u32,t:f32,s:f32,z:f32,w:f32 }; @group(0) @binding(BIND) var<uniform> p:P;`;
export function shader(buffers, body) {
  if (body.includes("global_invocation_id"))
    body = body.replaceAll("g.x", "(g.x+g.y*16776960u)");
  else if (body.includes("@workgroup_size(256)"))
    body = body.replaceAll("g.x", "(g.x+g.y*65535u)");
  return (
    buffers
      .map(
        (v, i) =>
          `@group(0) @binding(${i}) var<storage,${v[1] ?? "read"}> ${v[0]}:array<${v[2] ?? "f32"}>;`,
      )
      .join("\n") +
    "\n" +
    params.replace("BIND", buffers.length) +
    "\n" +
    body
  );
}
export function linear(dtype, useF16) {
  const half = dtype === "F16" && useF16;
  const fetch =
    dtype === "F32"
      ? "return bitcast<f32>(w[i]);"
      : dtype === "F16"
        ? half
          ? "return f32(w[i]);"
          : "let v=unpack2x16float(w[i/2u]);return select(v.x,v.y,i%2u==1u);"
        : "let v=i32((w[i/4u]>>((i%4u)*8u))&255u);return f32(select(v,v-256,v>=128))*scale[o];";
  return (
    (half ? "enable f16;\n" : "") +
    shader(
      [
        ["x"],
        ["w", "read", half ? "f16" : "u32"],
        ["bias"],
        ["scale"],
        ["y", "read_write"],
      ],
      `
 fn weight(i:u32,o:u32)->f32{${fetch}}
 var<workgroup> ta:array<f32,256>;var<workgroup> tb:array<f32,1024>;
 @compute @workgroup_size(8,8) fn main(@builtin(local_invocation_id) l:vec3<u32>,@builtin(workgroup_id) g:vec3<u32>){
  let tid=l.y*8u+l.x;let row=g.y*8u+l.y;let col=g.x*32u+l.x*4u;var acc=vec4<f32>(0.0);
  for(var start=0u;start<p.k;start+=32u){
   for(var j=tid;j<256u;j+=64u){let r=g.y*8u+j/32u;let k=start+j%32u;var v=0.0;if(r<p.n&&k<p.k){v=x[r*p.k+k];}ta[j]=v;}
   for(var j=tid;j<1024u;j+=64u){let k=start+j/32u;let o=g.x*32u+j%32u;var v=0.0;if(k<p.k&&o<p.c){v=weight(o*p.k+k,o);}tb[j]=v;}
   workgroupBarrier();
   for(var k=0u;k<32u;k++){let a=ta[l.y*32u+k];let i=k*32u+l.x*4u;acc+=a*vec4<f32>(tb[i],tb[i+1u],tb[i+2u],tb[i+3u]);}
   workgroupBarrier();
  }
  if(row<p.n){for(var j=0u;j<4u;j++){if(col+j<p.c){var b=0.0;if(p.a!=0u){b=bias[col+j];}y[row*p.c+col+j]=acc[j]+b;}}}
 }`,
    )
  );
}
export const norm = shader(
  [["x"], ["w"], ["b"], ["y", "read_write"]],
  `
 var<workgroup> red:array<f32,256>;var<workgroup> mean:f32;var<workgroup> inv:f32;
 @compute @workgroup_size(256) fn main(@builtin(local_invocation_index) t:u32,@builtin(workgroup_id) g:vec3<u32>){
  let base=g.x*p.c;var sum=0.0;for(var c=t;c<p.c;c+=256u){sum+=x[base+c];}red[t]=sum;workgroupBarrier();
  for(var s=128u;s>0u;s/=2u){if(t<s){red[t]+=red[t+s];}workgroupBarrier();}
  if(t==0u){mean=red[0]/f32(p.c);}workgroupBarrier();
  sum=0.0;for(var c=t;c<p.c;c+=256u){let d=x[base+c]-mean;sum+=d*d;}red[t]=sum;workgroupBarrier();
  for(var s=128u;s>0u;s/=2u){if(t<s){red[t]+=red[t+s];}workgroupBarrier();}
  if(t==0u){inv=inverseSqrt(red[0]/f32(p.c)+0.00001);}workgroupBarrier();
  for(var c=t;c<p.c;c+=256u){var v=(x[base+c]-mean)*inv;if(p.a!=0u){v=v*w[c]+b[c];}y[base+c]=v;}
 }`,
);
export const element = shader(
  [["x"], ["a"], ["b"], ["y", "read_write"]],
  `
 fn sigmoid(v:f32)->f32{return 1.0/(1.0+exp(-v));}
 @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>){let i=g.x;if(i>=p.n*p.c){return;}
 if(p.a==0u){y[i]=x[i]+a[i];}
 else if(p.a==1u){y[i]=x[i]*sigmoid(a[i%p.m])+b[i%p.m];}
 else if(p.a==2u){y[i]=x[i]+a[i]*sigmoid(b[i%p.m]);}
 else if(p.a==3u){let r=i/p.c;let c=i%p.c;let gate=x[r*p.c*2u+p.c+c];y[i]=x[r*p.c*2u+c]*(gate*sigmoid(gate));}
 }`,
);
export const rearrange = shader(
  [["x"], ["extra"], ["y", "read_write"]],
  `
 @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>){let i=g.x;if(i>=p.n*p.c){return;}let r=i/p.c;let c=i%p.c;
 if(p.a==0u){y[i]=x[r*p.k+p.b+c];}
 else if(p.a==1u){if(i<p.k){y[i]=x[i];}else{y[i]=extra[i-p.k];}}
 else if(p.a==2u){if(r<10u){y[i]=0.0;}else{y[i]=x[c];}}
 else if(p.a==3u){if(r<10u){y[i]=extra[i];}else{y[i]=x[(r-10u)*p.c+c];}}
 else if(p.a==4u){let ii=r/p.k;let jj=r%p.k;if(ii<10u||jj<10u){y[i]=0.0;}else{y[i]=x[((ii-10u)*(p.k-10u)+jj-10u)*p.c+c];}}
 else if(p.a==5u){y[i]=x[(r+10u)*p.c+c];}
 }`,
);
export const time = shader(
  [["freq"], ["y", "read_write"]],
  `
 @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>){let i=g.x;if(i<128u){let a=(p.t*2000.0)*freq[i];y[i]=sin(a);y[i+128u]=cos(a);}}`,
);
export const fold = shader(
  [["c"], ["a"], ["t"], ["y", "read_write"]],
  `
 @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>){let i=g.x;if(i<256u){y[i]=c[p.a*256u+i];y[i+256u]=a[p.b*256u+i];y[i+512u]=t[p.d*256u+i];}}`,
);
export const features = shader(
  [["sc"], ["indices"], ["breaks"], ["divisor"], ["y", "read_write"]],
  `
 @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>){let i=g.x;if(i>=p.n*132u){return;}let r=i/132u;let c=i%132u;
 if(c<128u){let index=indices[r]-indices[0]+1.0;let a=(index*3.141592653589793)/divisor[c%64u];if(c<64u){y[i]=sin(a);}else{y[i]=cos(a);}}
 else if(c<131u){y[i]=sc[r*3u+c-128u];}else{y[i]=breaks[r];}
 }`,
);
export const pairs = shader(
  [["x"], ["sc"], ["indices"], ["bins"], ["y", "read_write", "u32"]],
  `
 fn bucket(i:u32,j:u32,selfcond:bool)->u32{var sum=0.0;for(var k=0u;k<3u;k++){var d=x[i*3u+k]-x[j*3u+k];if(selfcond){d=sc[i*3u+k]-sc[j*3u+k];}sum+=d*d;}let dist=sqrt(sum);var count=63u;var offset=127u;if(selfcond){count=127u;offset=0u;}var b=0u;for(var k=0u;k<count;k++){if(dist>bins[offset+k]){b++;}}return b;}
 @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>){let r=g.x;if(r>=p.n*p.n){return;}let i=r/p.n;let j=r%p.n;
 y[r*3u]=u32(clamp(ceil(indices[i]-indices[j]+62.5),0.0,126.0));y[r*3u+1u]=0xffffffffu;if(p.a==1u){y[r*3u+1u]=127u+bucket(i,j,true);}y[r*3u+2u]=255u+bucket(i,j,false);
 }`,
);
export function sparseLinear(dtype, useF16) {
  const half = dtype === "F16" && useF16;
  const fetch =
    dtype === "F32"
      ? "return bitcast<f32>(w[i]);"
      : dtype === "F16"
        ? half
          ? "return f32(w[i]);"
          : "let v=unpack2x16float(w[i/2u]);return select(v.x,v.y,i%2u==1u);"
        : "let v=i32((w[i/4u]>>((i%4u)*8u))&255u);return f32(select(v,v-256,v>=128))*scale[o];";
  return (
    (half ? "enable f16;\n" : "") +
    shader(
      [
        ["indices", "read", "u32"],
        ["w", "read", half ? "f16" : "u32"],
        ["scale"],
        ["y", "read_write"],
      ],
      `
 fn weight(i:u32,o:u32)->f32{${fetch}}
 @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>){let i=g.x;if(i>=p.n*p.c){return;}let r=i/p.c;let c=i%p.c;var s=0.0;for(var j=0u;j<3u;j++){let idx=indices[r*3u+j];if(idx!=0xffffffffu){s+=weight(c*p.k+idx,c);}}y[i]=s;}
 `,
    )
  );
}
export const logits = shader(
  [["q"], ["k"], ["bias"], ["y", "read_write"]],
  `
 @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>){let idx=g.x;if(idx>=p.n*p.n*12u){return;}let j=idx%p.n;let i=(idx/p.n)%p.n;let h=idx/(p.n*p.n);var s=0.0;
 for(var d=0u;d<64u;d++){s+=q[i*768u+h*64u+d]*k[j*768u+h*64u+d];}y[idx]=s*0.125+bias[(i*p.n+j)*12u+h];}`,
);
export const softmax = shader(
  [["x"], ["y", "read_write"]],
  `
 var<workgroup> red:array<f32,256>;var<workgroup> mx:f32;var<workgroup> total:f32;
 @compute @workgroup_size(256) fn main(@builtin(local_invocation_index) t:u32,@builtin(workgroup_id) g:vec3<u32>){let base=g.x*p.c;var m=-3.402823e38;
 for(var j=t;j<p.c;j+=256u){m=max(m,x[base+j]);}red[t]=m;workgroupBarrier();for(var s=128u;s>0u;s/=2u){if(t<s){red[t]=max(red[t],red[t+s]);}workgroupBarrier();}if(t==0u){mx=red[0];}workgroupBarrier();
 var sum=0.0;for(var j=t;j<p.c;j+=256u){sum+=exp(x[base+j]-mx);}red[t]=sum;workgroupBarrier();for(var s=128u;s>0u;s/=2u){if(t<s){red[t]+=red[t+s];}workgroupBarrier();}if(t==0u){total=red[0];}workgroupBarrier();
 for(var j=t;j<p.c;j+=256u){y[base+j]=exp(x[base+j]-mx)/total;}}
 `,
);
export const attend = shader(
  [["prob"], ["v"], ["gate"], ["y", "read_write"]],
  `
 @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>){let idx=g.x;if(idx>=p.n*768u){return;}let i=idx/768u;let h=(idx%768u)/64u;let d=idx%64u;var s=0.0;
 for(var j=0u;j<p.n;j++){s+=prob[(h*p.n+i)*p.n+j]*v[j*768u+h*64u+d];}y[idx]=s/(1.0+exp(-gate[idx]));}`,
);
