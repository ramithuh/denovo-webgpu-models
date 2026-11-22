import schedule from "./diffusion-constants.js";
import { rigidFrames, normLast } from "./geometry.js";
export function rotationVector(o, R) {
  const L = R.shape[0];
  const js = `const b=Math.floor(i/3)*9,k=i%3;const r=Array.from(a0.slice(b,b+9));let q=[0,0,0,0];const trace=r[0]+r[4]+r[8];if(trace>0){const s=Math.sqrt(1+trace)*2;q=[(r[7]-r[5])/s,(r[2]-r[6])/s,(r[3]-r[1])/s,s/4];}else{let d=0;if(r[4]>r[0])d=1;if(r[8]>r[d*3+d])d=2;const j=(d+1)%3,l=(d+2)%3,s=2*Math.sqrt(1+r[d*3+d]-r[j*3+j]-r[l*3+l]);q[d]=s/4;q[j]=(r[d*3+j]+r[j*3+d])/s;q[l]=(r[d*3+l]+r[l*3+d])/s;q[3]=(r[l*3+j]-r[j*3+l])/s;}if(q[3]<0)q=q.map(x=>-x);const n=Math.hypot(q[0],q[1],q[2]),a=2*Math.atan2(n,q[3]);o[i]=n>1e-8?q[k]*a/n:2*q[k];`;
  const wg = `let b=(i/3u)*9u;let k=i%3u;var r:array<f32,9>;for(var j=0u;j<9u;j++){r[j]=a0[b+j];}var q:array<f32,4>;let tr=r[0]+r[4]+r[8];if(tr>0.0){let s=2.0*sqrt(1.0+tr);q=array<f32,4>((r[7]-r[5])/s,(r[2]-r[6])/s,(r[3]-r[1])/s,s/4.0);}else{var d=0u;if(r[4]>r[0]){d=1u;}if(r[8]>r[d*3u+d]){d=2u;}let j=(d+1u)%3u;let l=(d+2u)%3u;let s=2.0*sqrt(1.0+r[d*3u+d]-r[j*3u+j]-r[l*3u+l]);q[d]=s/4.0;q[j]=(r[d*3u+j]+r[j*3u+d])/s;q[l]=(r[d*3u+l]+r[l*3u+d])/s;q[3]=(r[l*3u+j]-r[j*3u+l])/s;}if(q[3]<0.0){for(var j=0u;j<4u;j++){q[j]=-q[j];}}let n=length(vec3<f32>(q[0],q[1],q[2]));let a=2.0*atan2(n,q[3]);var v=2.0*q[k];if(n>1e-8){v=q[k]*a/n;}o[i]=v;`;
  return o.kernel([L, 3], [R], js, wg);
}
export function rotationExp(o, v) {
  const L = v.shape[0],
    js = `const b=Math.floor(i/9)*3,k=i%9,x=a0[b],y=a0[b+1],z=a0[b+2],a=Math.sqrt(x*x+y*y+z*z),s=a>1e-5?Math.sin(a)/a:1-a*a/6,c=a>1e-5?(1-Math.cos(a))/(a*a):.5-a*a/24;const r=[1-c*(y*y+z*z),c*x*y-s*z,c*x*z+s*y,c*x*y+s*z,1-c*(x*x+z*z),c*y*z-s*x,c*x*z-s*y,c*y*z+s*x,1-c*(x*x+y*y)];o[i]=r[k];`,
    wg = `let b=(i/9u)*3u;let k=i%9u;let x=a0[b];let y=a0[b+1u];let z=a0[b+2u];let a=sqrt(x*x+y*y+z*z);var s=1.0-a*a/6.0;var c=0.5-a*a/24.0;if(a>1e-5){s=sin(a)/a;c=(1.0-cos(a))/(a*a);}var r=array<f32,9>(1.0-c*(y*y+z*z),c*x*y-s*z,c*x*z+s*y,c*x*y+s*z,1.0-c*(x*x+z*z),c*y*z-s*x,c*x*z-s*y,c*y*z+s*x,1.0-c*(x*x+y*y));o[i]=r[k];`;
  return o.kernel([L, 3, 3], [v], js, wg);
}
export function rotationUpdate(o, xt, px0, t, noise) {
  const L = xt.shape[0],
    rt = rigidFrames(o, xt).R,
    r0 = rigidFrames(o, px0).R,
    log = rotationVector(o, o.einsum("nij,nkj->nik", rt, r0)),
    omega = normLast(o, log),
    table = o.tensor([1000], schedule.score[t - 1]);
  const scale = o.kernel(
      [L, 1],
      [omega, table],
      `const a=a0[i],q=Math.max(0,Math.min(999,a*1000/Math.PI-1)),lo=Math.floor(q),hi=Math.min(lo+1,999),s=a1[lo]+(a1[hi]-a1[lo])*(q-lo);o[i]=a>1e-8?s/a:0;`,
      `let a=a0[i];let q=clamp(a*1000.0/3.141592653589793-1.0,0.0,999.0);let lo=u32(floor(q));let hi=min(lo+1u,999u);let s=mix(a1[lo],a1[hi],q-f32(lo));var v=0.0;if(a>1e-8){v=s/a;}o[i]=v;`,
    ),
    g = schedule.g[t - 1];
  return rotationExp(
    o,
    o.add(
      o.mul(o.mul(log, scale), Math.fround(g * g) / 50),
      o.mul(noise, g * Math.sqrt(1 / 50)),
    ),
  );
}
export function reverseStep(
  o,
  { xt, px0, t, translationNoise, rotationNoise, mask, rotationUpdates },
) {
  const L = xt.shape[0];
  if (t < 2 || t > 50 || !Number.isInteger(t))
    throw Error("Reverse update t must be in [2,50]");
  const cA = (x) => o.reshape(o.slice(x, 1, 1, 2), [L, 3]),
    ca = cA(xt),
    pred = cA(px0),
    movable = o.tensor(
      [L, 1],
      Array.from(mask, (x) => 1 - Number(x)),
    ),
    scaledCA = o.mul(ca, 0.25),
    scaledPred = o.mul(pred, 0.25),
    mu = o.add(
      o.mul(scaledPred, schedule.coef0[t - 1]),
      o.mul(scaledCA, schedule.coefT[t - 1]),
    ),
    sampled = o.add(
      mu,
      o.mul(translationNoise, schedule.translationStd[t - 1]),
    ),
    delta = o.mul(o.div(o.sub(sampled, scaledCA), 0.25), movable);
  let rotation =
    rotationUpdates ?? rotationUpdate(o, xt, px0, t, rotationNoise);
  const identity = o.tensor([1, 3, 3], [1, 0, 0, 0, 1, 0, 0, 0, 1]);
  rotation = o.add(
    o.mul(rotation, o.reshape(movable, [L, 1, 1])),
    o.mul(identity, o.reshape(o.sub(1, movable), [L, 1, 1])),
  );
  const frames = o.add(
      o.einsum("nij,naj->nai", rotation, o.sub(xt, o.reshape(ca, [L, 1, 3]))),
      o.reshape(ca, [L, 1, 3]),
    ),
    crds = o.add(frames, o.reshape(delta, [L, 1, 3]));
  return { coords: crds, rotation, delta };
}
