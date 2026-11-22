// Genie 2 frame geometry. Adapted from the pinned Apache-2.0 implementation.
import { portable } from "./portable.js";
export function frenet(o, x, chains, mask) {
  const L = x.shape[0];
  const tangents = portable(
    o,
    [L - 1, 3],
    [x],
    `
    let r=u32(floor(f32(i)/3.0));let c=i%3u;var norm=1e-10;
    for(var k=0u;k<3u;k++){let d=a0[(r+1u)*3u+k]-a0[r*3u+k];norm=norm+d*d;}
    o[i]=rfdiv(a0[(r+1u)*3u+c]-a0[r*3u+c],sqrt(norm));`,
  );
  const interior = portable(
    o,
    [L - 2, 9],
    [tangents],
    `
    let r=u32(floor(f32(i)/9.0));let row=u32(floor(f32(i%9u)/3.0));let col=i%3u;
    var b:array<f32,3>;var norm=1e-10;
    for(var k=0u;k<3u;k++){let j=(k+1u)%3u;let h=(k+2u)%3u;b[k]=a0[r*3u+j]*a0[(r+1u)*3u+h]-a0[r*3u+h]*a0[(r+1u)*3u+j];norm=norm+b[k]*b[k];}
    for(var k=0u;k<3u;k++){b[k]=rfdiv(b[k],sqrt(norm));}
    if(col==0u){o[i]=a0[(r+1u)*3u+row];}
    else if(col==1u){o[i]=b[row];}
    else {let j=(row+1u)%3u;let h=(row+2u)%3u;o[i]=b[j]*a0[(r+1u)*3u+h]-b[h]*a0[(r+1u)*3u+j];}`,
  );
  return portable(
    o,
    [L, 3, 3],
    [interior, chains, mask],
    `
    let r=u32(floor(f32(i)/9.0));let c=i%9u;var n=0u;for(var k=0u;k<${L}u;k++){n=n+u32(a2[k]);}
    if(r>=n){o[i]=0.0;if(c==0u||c==4u||c==8u){o[i]=1.0;}}
    else {var j=r;if(r==0u){j=1u;}else if(a1[r]!=a1[r-1u]){j=r+1u;}
      if(r==n-1u){j=r-1u;}else if(a1[r]!=a1[r+1u]){j=r-1u;}
      j=min(max(j,1u),n-2u);o[i]=a0[(j-1u)*9u+c];}`,
  );
}
export function applyPoints(o, raw, rots, trans, heads, points) {
  const L = trans.shape[0],
    W = heads * points;
  return portable(
    o,
    [L, heads, points, 3],
    [raw, rots, trans],
    `
    let r=u32(floor(f32(i)/${W * 3}.0));let p=u32(floor(f32(i%${W * 3}u)/3.0));let c=i%3u;
    var v=a1[r*9u+c*3u]*a0[r*${W * 3}u+p];
    v=v+a1[r*9u+c*3u+1u]*a0[r*${W * 3}u+${W}u+p];
    v=v+a1[r*9u+c*3u+2u]*a0[r*${W * 3}u+${2 * W}u+p];o[i]=v+a2[r*3u+c];`,
  );
}
export function invertPoints(o, p, rots, trans) {
  const L = trans.shape[0],
    W = p.size / L / 3;
  return portable(
    o,
    p.shape,
    [p, rots, trans],
    `
    let r=u32(floor(f32(i)/${W * 3}.0));let base=i-i%3u;let c=i%3u;
    var v=a1[r*9u+c]*(a0[base]-a2[r*3u]);
    v=v+a1[r*9u+3u+c]*(a0[base+1u]-a2[r*3u+1u]);
    v=v+a1[r*9u+6u+c]*(a0[base+2u]-a2[r*3u+2u]);o[i]=v;`,
  );
}
export function backboneUpdate(o, params, rots, trans) {
  const L = trans.shape[0];
  const delta = portable(
    o,
    [L, 3, 3],
    [params],
    `
    let r=u32(floor(f32(i)/9.0));let c=i%9u;var q:array<f32,4>;
    q[0]=1.0;q[1]=a0[r*6u];q[2]=a0[r*6u+1u];q[3]=a0[r*6u+2u];
    let norm=sqrt(((q[1]*q[1]+q[2]*q[2])+q[3]*q[3])+1.0);
    for(var k=0u;k<4u;k++){q[k]=rfdiv(q[k],norm);}
    let a=q[0];let b=q[1];let c2=q[2];let d=q[3];var v=0.0;
    if(c==0u){v=((a*a+b*b)-c2*c2)-d*d;}
    if(c==1u){v=-2.0*a*d+2.0*b*c2;}
    if(c==2u){v=2.0*a*c2+2.0*b*d;}
    if(c==3u){v=2.0*a*d+2.0*b*c2;}
    if(c==4u){v=((a*a-b*b)+c2*c2)-d*d;}
    if(c==5u){v=-2.0*a*b+2.0*c2*d;}
    if(c==6u){v=-2.0*a*c2+2.0*b*d;}
    if(c==7u){v=2.0*a*b+2.0*c2*d;}
    if(c==8u){v=((a*a-b*b)-c2*c2)+d*d;}o[i]=v;`,
  );
  const r = portable(
    o,
    [L, 3, 3],
    [rots, delta],
    `
    let r=u32(floor(f32(i)/9.0));let a=u32(floor(f32(i%9u)/3.0));let b=i%3u;
    o[i]=(a0[r*9u+a*3u]*a1[r*9u+b]+a0[r*9u+a*3u+1u]*a1[r*9u+3u+b])+a0[r*9u+a*3u+2u]*a1[r*9u+6u+b];`,
  );
  const t = portable(
    o,
    [L, 3],
    [rots, trans, params],
    `
    let r=u32(floor(f32(i)/3.0));let c=i%3u;
    o[i]=((a0[r*9u+c*3u]*a2[r*6u+3u]+a0[r*9u+c*3u+1u]*a2[r*6u+4u])+a0[r*9u+c*3u+2u]*a2[r*6u+5u])+a1[i];`,
  );
  return { rots: r, trans: t };
}
