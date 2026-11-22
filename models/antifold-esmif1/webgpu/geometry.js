import {decl} from './shaders.js';
const helpers=`
fn finite(v:f32)->f32{return select(0.0,v,abs(v)<=3.402823e38);}
fn atom(i:u32)->vec3<f32>{return vec3<f32>(finite(coords[i*3u]),finite(coords[i*3u+1u]),finite(coords[i*3u+2u]));}
fn unit(v:vec3<f32>,kind:u32)->vec3<f32>{var den=sqrt(dot(v,v)+0.00000001);if(kind==1u){den=max(sqrt(dot(v,v)),0.000000000001);}return v/den;}
fn angle(j:u32,kind:u32)->f32{
 let a=unit(atom(j+1u)-atom(j),kind);let b=unit(atom(j+2u)-atom(j+1u),kind);let c=unit(atom(j+3u)-atom(j+2u),kind);
 let n2=unit(cross(a,b),kind);let n1=unit(cross(b,c),kind);
 return sign(dot(a,n1))*acos(clamp(dot(n2,n1),-0.9999999,0.9999999));}
`;
export function nodeGeometry(N){return `${decl(0,'coords')}${decl(1,'mask','i32')}${decl(2,'confidence')}${decl(3,'y','f32',true)}${helpers}
 @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){let r=g.x;if(r>=${N}u){return;}let off=r*47u;
 for(var j=0u;j<3u;j++){let k=r*3u+j;var a=0.0;var b=0.0;if(k>=1u&&k<${N*3-2}u){a=angle(k-1u,0u);b=angle(k-1u,1u);}y[off+j]=cos(a);y[off+3u+j]=sin(a);y[off+16u+j]=cos(b);y[off+19u+j]=sin(b);}
 y[off+6u]=f32(mask[r]);
 let origin=atom(r*3u+1u);let c=atom(r*3u+2u)-origin;let n=atom(r*3u)-origin;
 let cn=unit(c,0u);let nn=unit(n,0u);let bis=unit(cn+nn,0u);let perp=unit(cross(cn,nn),0u);
 let side=-bis*0.5773502691896258-perp*0.816496580927726;
 var forward=vec3<f32>(0.0);var backward=vec3<f32>(0.0);
 if(r+1u<${N}u){forward=unit(atom((r+1u)*3u+1u)-origin,0u);}if(r>0u){backward=unit(atom((r-1u)*3u+1u)-origin,0u);}
 let e1=cn;let e2=unit(n-e1*dot(e1,n),0u);let e3=cross(e1,e2);
 for(var j=0u;j<3u;j++){y[off+7u+j]=forward[j];y[off+10u+j]=backward[j];y[off+13u+j]=side[j];y[off+22u+j]=e1[j];y[off+25u+j]=e2[j];y[off+28u+j]=e3[j];}
 for(var j=0u;j<16u;j++){let z=(confidence[r]-f32(j)/15.0)*16.0;y[off+31u+j]=exp(-z*z);}
 }`;
}
export function neighbors(N,K){return `${decl(0,'coords')}${decl(1,'mask','i32')}${decl(2,'y','f32',true)}${helpers}
 @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){let src=g.x;if(src>=${N}u){return;}
 var ds:array<f32,30>;var ids:array<i32,30>;for(var j=0u;j<${K}u;j++){ds[j]=3.402823e38;ids[j]=-1;}
 if(mask[src]!=0){for(var dst=0u;dst<${N}u;dst++){if(mask[dst]==0){continue;}let v=atom(src*3u+1u)-atom(dst*3u+1u);let d=sqrt(dot(v,v)+0.00000001);var pos=${K}u;
 for(var j=0u;j<${K}u;j++){if(d<ds[j]){pos=j;break;}}
 if(pos<${K}u){for(var j=${K-1}u;j>pos;j--){ds[j]=ds[j-1u];ids[j]=ids[j-1u];}ds[pos]=d;ids[pos]=i32(dst);}}}
 for(var j=0u;j<${K}u;j++){let off=(src*${K}u+j)*2u;y[off]=bitcast<f32>(select(-1,i32(src),ids[j]>=0));y[off+1u]=bitcast<f32>(ids[j]);}}
 `;
}
export function edgeGeometry(E){return `${decl(0,'coords')}${decl(1,'indices','i32')}${decl(2,'y','f32',true)}${helpers}
 @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g:vec3<u32>){let e=g.x;if(e>=${E}u){return;}let off=e*37u;
 let src=indices[e*2u];let dst=indices[e*2u+1u];if(src<0||dst<0){for(var j=0u;j<37u;j++){y[off+j]=0.0;}return;}
 let v=atom(u32(src)*3u+1u)-atom(u32(dst)*3u+1u);let d=sqrt(dot(v,v)+0.00000001);
 for(var j=0u;j<16u;j++){let z=(d-f32(j)*20.0/15.0)/1.25;y[off+j]=exp(-z*z);}
 for(var j=0u;j<8u;j++){let a=f32(src-dst)*exp(f32(j*2u)*(-0.5756462732485115));y[off+16u+j]=cos(a);y[off+24u+j]=sin(a);}
 y[off+32u]=0.0;y[off+33u]=0.0;let vn=unit(v,0u);for(var j=0u;j<3u;j++){y[off+34u+j]=vn[j];}}
 `;
}
