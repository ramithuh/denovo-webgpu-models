// Pure WGSL compute. Storage precision never changes activation/accumulation F32.
export const decl=(i,name,type='f32',rw=false)=>`@group(0) @binding(${i}) var<storage,${rw?'read_write':'read'}> ${name}:array<${type}>;\n`;
export function element(n,code,types=['f32']) {
 return types.map((t,i)=>decl(i,`x${i}`,t)).join('')+decl(types.length,'y','f32',true)+`
 @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3<u32>){let i=g.x+g.y*16776960u;if(i<${n}u){${code}}}`;
}
export function matmul(M,N,K,dtype='F32',hasBias=false,useF16=false){
 const native=dtype==='F16'&&useF16;
 const get=dtype==='F32'?'return w[i];':dtype==='F16'?(native?'return f32(w[i]);':'return unpack2x16float(w[i/2u])[i%2u];'):'let q=i32((w[i/4u]>>((i%4u)*8u))&255u);return f32(select(q,q-256,q>=128))*sc[row];';
 return `${native?'enable f16;':''}${decl(0,'a')}${decl(1,'w',dtype==='F32'?'f32':native?'f16':'u32')}${decl(2,'b')}${decl(3,'sc')}${decl(4,'y','f32',true)}
 fn weight(i:u32,row:u32)->f32{${get}}
 var<workgroup> at:array<f32,256>;var<workgroup> bt:array<f32,256>;
 @compute @workgroup_size(16,16) fn main(@builtin(local_invocation_id) t:vec3<u32>,@builtin(workgroup_id) g:vec3<u32>){
 let r=g.y*16u+t.y;let c=g.x*16u+t.x;let idx=t.y*16u+t.x;var s=0.0;
 for(var base=0u;base<${K}u;base+=16u){var av=0.0;var bv=0.0;
 if(r<${M}u&&base+t.x<${K}u){av=a[r*${K}u+base+t.x];}
 if(c<${N}u&&base+t.y<${K}u){bv=weight(c*${K}u+base+t.y,c);}
 at[idx]=av;bt[idx]=bv;workgroupBarrier();
 for(var k=0u;k<16u;k++){s=s+at[t.y*16u+k]*bt[k*16u+t.x];}workgroupBarrier();}
 if(r<${M}u&&c<${N}u){y[r*${N}u+c]=s${hasBias?'+b[c]':''};}}
 `;
}
export function layerNorm(M,D,dihedral=false){return `${decl(0,'x')}${decl(1,'w')}${decl(2,'b')}${decl(3,'y','f32',true)}
 var<workgroup> tmp:array<f32,256>;
 @compute @workgroup_size(256) fn main(@builtin(local_invocation_index) t:u32,@builtin(workgroup_id) g:vec3<u32>){
 let row=g.x*${D}u;var s=0.0;for(var j=t;j<${D}u;j+=256u){s+=x[row+j];}tmp[t]=s;workgroupBarrier();
 for(var k=128u;k>0u;k/=2u){if(t<k){tmp[t]+=tmp[t+k];}workgroupBarrier();}let mean=tmp[0]/${D}.0;var v=0.0;
 for(var j=t;j<${D}u;j+=256u){let d=x[row+j]-mean;v+=d*d;}workgroupBarrier();tmp[t]=v;workgroupBarrier();
 for(var k=128u;k>0u;k/=2u){if(t<k){tmp[t]+=tmp[t+k];}workgroupBarrier();}
 let den=${dihedral?`sqrt(tmp[0]/${D-1}.0+0.000001)+0.000001`:`sqrt(tmp[0]/${D}.0+0.00001)`};
 for(var j=t;j<${D}u;j+=256u){y[row+j]=(x[row+j]-mean)/den*w[j]+b[j];}}`;
}
export function vectorNorm(M,D){const V=D/3;return `${decl(0,'x')}${decl(1,'y','f32',true)}
 @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){let r=g.x;if(r>=${M}u){return;}var sum=0.0;var count=0.0;
 for(var j=0u;j<${V}u;j++){let i=r*${D}u+j*3u;let v=vec3<f32>(x[i],x[i+1u],x[i+2u]);let n=dot(v,v)+0.0001;if(n>0.0002){sum+=n;count+=1.0;}}
 let den=sqrt(sum/(0.0001+count)+0.0001);
 for(var j=0u;j<${V}u;j++){let i=r*${D}u+j*3u;let v=vec3<f32>(x[i],x[i+1u],x[i+2u]);let mask=select(0.0,1.0,dot(v,v)+0.0001>0.0002);y[i]=mask*x[i]/den;y[i+1u]=mask*x[i+1u]/den;y[i+2u]=mask*x[i+2u]/den;}}`;
}
export function scores(M,S,D,causal){const C=D/8;return element(8*M*S,`let col=i%${S}u;let r=(i/${S}u)%${M}u;let h=i/${M*S}u;var s=0.0;for(var j=0u;j<${C}u;j++){s+=x0[r*${D}u+h*${C}u+j]*x1[col*${D}u+h*${C}u+j];}y[i]=select(s*${1/Math.sqrt(C)},-3.402823e38,x2[col]!=0${causal?'||col>r':''});`,['f32','f32','i32']);}
export function softmax(R,S){return `${decl(0,'x')}${decl(1,'y','f32',true)}
 var<workgroup> t:array<f32,256>;
 @compute @workgroup_size(256) fn main(@builtin(local_invocation_index) j:u32,@builtin(workgroup_id) g:vec3<u32>){let off=g.x*${S}u;var m=-3.402823e38;
 for(var k=j;k<${S}u;k+=256u){m=max(m,x[off+k]);}t[j]=m;workgroupBarrier();
 for(var n=128u;n>0u;n/=2u){if(j<n){t[j]=max(t[j],t[j+n]);}workgroupBarrier();}m=t[0];
 var s=0.0;for(var k=j;k<${S}u;k+=256u){s+=exp(x[off+k]-m);}workgroupBarrier();t[j]=s;workgroupBarrier();
 for(var n=128u;n>0u;n/=2u){if(j<n){t[j]+=t[j+n];}workgroupBarrier();}let den=t[0];
 for(var k=j;k<${S}u;k+=256u){y[off+k]=exp(x[off+k]-m)/den;}}`;
}
