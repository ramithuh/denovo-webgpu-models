import { abort } from './loader.js';
import { fmaF32 } from './fma.js';
import { sinCosF32 } from './numeric.js';
const f=Math.fround;
export function schedule(steps,mode,p=2) {
  if(!Number.isInteger(steps)||steps<1)throw Error('steps must be a positive integer');
  const t=new Float32Array(steps+1);
  for(let i=0;i<=steps;i++){
    // PyTorch linspace/logspace use the nearer endpoint to avoid accumulated drift.
    const u=i<=(steps/2)?f(i*f(1/steps)):fmaF32(-(steps-i),f(1/steps),1);
    if(mode==='uniform')t[i]=u;
    else if(mode==='power')t[i]=f(u**p);
    else if(mode==='log'){
      const j=steps-i,step=p/steps,e=j<=(steps/2)?-p+j*step:-(steps-j)*step;
      t[i]=f(1-f(10**e));
    }else throw Error(`Unknown schedule ${mode}`);
  }
  if(mode==='log'){const max=t[steps];for(let i=0;i<=steps;i++)t[i]=f(t[i]/max);}
  return t;
}
export function noiseSchedule(t,mode) {
  t=f(Math.max(0,Math.min(1-1e-5,t)));
  if(mode==='1/t')return f(1/f(t+f(.01)));
  if(mode==='tan'){
    const angle=f(f(f(1-t)*f(Math.PI))/2);
    const [sn,cs]=sinCosF32(angle,false);
    return f(f(f(Math.PI/2)*sn)/f(cs+f(.01)));
  }
  throw Error(`Unknown noise schedule ${mode}`);
}
export function randomNormal(seed=1) {
  let state=seed>>>0;
  const uniform=()=>{state=(state+0x6d2b79f5)>>>0;let x=state;x=Math.imul(x^(x>>>15),x|1);x^=x+Math.imul(x^(x>>>7),x|61);return ((x^(x>>>14))>>>0)/4294967296;};
  return n=>{const a=new Float32Array(n);for(let i=0;i<n;i+=2){const r=Math.sqrt(-2*Math.log(Math.max(1e-12,uniform()))),t=2*Math.PI*uniform();a[i]=r*Math.cos(t);if(i+1<n)a[i+1]=r*Math.sin(t);}return a;};
}
export function validateLength(length) {
  if(!Number.isInteger(length)||length<1||length>500)throw Error('length must be an integer in [1, 500]');
}
export function validateArray(x,n,name) {
  if(!x||x.length!==n||Array.from(x).some(v=>!Number.isFinite(v)))throw Error(`Invalid ${name}: expected ${n} finite numbers`);
  return x;
}
export function center(o,x) {
  const L=x.shape[0],D=x.shape[1];
  // PyTorch's scalar outer reduction uses four interleaved lanes and
  // 16-element cascade chunks. At L <= 500, two cascade levels suffice.
  const groups=Math.floor(L/4),chunks=Math.floor(groups/16),tail=groups*4;
  const sum=o.kernel([D],[x],
    `const f=Math.fround,lanes=new Float32Array(4);for(let lane=0;lane<4;lane++){let total=0;for(let ch=0;ch<${chunks};ch++){let s=0;for(let j=0;j<16;j++)s=f(s+a0[((ch*16+j)*4+lane)*${D}+i]);total=f(total+s);}let rest=0;for(let j=${chunks*16};j<${groups};j++)rest=f(rest+a0[(j*4+lane)*${D}+i]);lanes[lane]=f(rest+total);}for(let j=${tail};j<${L};j++)lanes[0]=f(lanes[0]+a0[j*${D}+i]);o[i]=f(f(f(lanes[0]+lanes[1])+lanes[2])+lanes[3]);`,
    `var lanes:array<f32,4>;for(var lane=0u;lane<4u;lane++){var total=0.0;for(var ch=0u;ch<${chunks}u;ch++){var s=0.0;for(var j=0u;j<16u;j++){s=s+a0[((ch*16u+j)*4u+lane)*${D}u+i];}total=total+s;}var rest=0.0;for(var j=${chunks*16}u;j<${groups}u;j++){rest=rest+a0[(j*4u+lane)*${D}u+i];}lanes[lane]=rest+total;}for(var j=${tail}u;j<${L}u;j++){lanes[0]=lanes[0]+a0[j*${D}u+i];}o[i]=((lanes[0]+lanes[1])+lanes[2])+lanes[3];`);
  return o.sub(x,o.div(o.reshape(sum,[1,D]),L));
}
export function integrate(o,x,v,noise,t,dt,g,mode,centered,noiseScale) {
  let delta;
  if(mode==='ode')delta=o.mul(v,dt);
  else {
    const score=o.div(o.sub(o.mul(v,t),x),f(1-t));
    if(t>f(.98)){
      const scaled=o.mul(score,1.5);
      const velocity=o.div(o.add(x,o.mul(scaled,f(1-t))),t);
      delta=o.mul(velocity,dt);
    }else{
      const std=f(Math.sqrt(f(f(f(2*g)*f(noiseScale))*dt)));
      delta=o.add(o.mul(o.add(v,o.mul(score,g)),dt),o.mul(noise,std));
    }
  }
  const next=o.add(x,delta);
  return centered?center(o,next):next;
}
export async function sample(o,ld,ae,options={}) {
  const {
    length=100,steps=400,seed=5,mode='sde',signal,onProgress,returnTrajectory=false,
    initialCa,initialZ,draws,noiseScaleCa=.1,noiseScaleZ=.1,motifState=null,
    centerEveryStep=!motifState,
  }=options;
  validateLength(length);
  if(!['sde','ode'].includes(mode))throw Error('mode must be sde or ode');
  for(const s of [noiseScaleCa,noiseScaleZ])if(!Number.isFinite(s)||s<0)throw Error('Noise scales must be nonnegative');
  const tCa=schedule(steps,'log',2),tZ=schedule(steps,'power',2),normal=randomNormal(seed),L=length;
  let ca=o.tensor([L,3],validateArray(initialCa??normal(L*3),L*3,'initialCa'));
  let z=o.tensor([L,8],validateArray(initialZ??normal(L*8),L*8,'initialZ'));
  let scCa,scZ,drawIndex=0;const trajectory=[];
  const motifKeep=motifState?.keep??[];
  const getDraw=(n)=>validateArray(draws?draws[drawIndex++]:normal(n),n,'noise draw');
  for(let step=0;step<steps;step++){
    abort(signal);
    const v=await ld.forward({ca,z,scCa,scZ,tCa:tCa[step],tZ:tZ[step],motifState},{signal,onProgress});
    if(returnTrajectory){
      if(o.readMany){const [rca,rz,rvca,rvz]=await o.readMany([ca,z,v.ca,v.z]);trajectory.push({ca:rca,z:rz,velocityCa:rvca,velocityZ:rvz});}
      else trajectory.push({ca:await o.read(ca),z:await o.read(z),velocityCa:await o.read(v.ca),velocityZ:await o.read(v.z)});
    }
    scCa=o.add(ca,o.mul(v.ca,f(1-tCa[step])));
    scZ=o.add(z,o.mul(v.z,f(1-tZ[step])));
    const epsCa=mode==='sde'&&tCa[step]<=f(.98)?o.tensor([L,3],getDraw(L*3)):null;
    const epsZ=mode==='sde'&&tZ[step]<=f(.98)?o.tensor([L,8],getDraw(L*8)):null;
    ca=integrate(o,ca,v.ca,epsCa,tCa[step],f(tCa[step+1]-tCa[step]),noiseSchedule(tCa[step],'1/t'),mode,centerEveryStep,noiseScaleCa);
    z=integrate(o,z,v.z,epsZ,tZ[step],f(tZ[step+1]-tZ[step]),noiseSchedule(tZ[step],'tan'),mode,false,noiseScaleZ);
    onProgress?.({phase:'sampling',step:step+1,total:steps});
    if(o.stepBoundary)await o.stepBoundary([ca,z,scCa,scZ,...motifKeep],step);
    else await o.checkpoint([ca,z,scCa,scZ,...motifKeep]);
  }
  if(draws&&drawIndex!==draws.length)throw Error(`Unused noise draws: ${draws.length-drawIndex}`);
  const dec=await ae.forward({ca,z},{signal,onProgress});
  const [rca,rz,rcoords,rlogits]=o.readMany?await o.readMany([ca,z,dec.coords,dec.logits]):[await o.read(ca),await o.read(z),await o.read(dec.coords),await o.read(dec.logits)];
  const result={ca:rca,z:rz,coords:rcoords,logits:rlogits,trajectory};
  abort(signal);return result;
}
