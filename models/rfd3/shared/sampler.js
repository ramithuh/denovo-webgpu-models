import { applySymmetry, validateSymmetry } from './symmetry.js';
import { rigidAlign, rigidAugment } from './alignment.js';
const f32=Math.fround;
export function noiseSchedule({steps=200,sigmaData=16,sMin=0.0004,sMax=160,p=7,partialT}={}) {
  if(!Number.isInteger(steps)||steps<2||![sigmaData,sMin,sMax,p].every(v=>Number.isFinite(v)&&v>0)||sMax<=sMin)
    throw new RangeError('Invalid diffusion schedule');
  if(partialT!==undefined&&(!Number.isFinite(partialT)||partialT<=0))throw new RangeError('partialT must be positive');
  const start=sMax**(1/p),delta=sMin**(1/p)-start;
  let values=Float32Array.from({length:steps},(_,i)=>f32(sigmaData*f32(f32(f32(start)+f32(f32(i/(steps-1))*f32(delta)))**p)));
  if(partialT!==undefined)values=values.filter(v=>v<=partialT);
  if(values.length<2)throw new RangeError('Partial diffusion requires at least two schedule points');
  return values;
}
export function resolveRandom(seed) {
  const resolved=seed??crypto.getRandomValues(new Uint32Array(1))[0];
  if(!Number.isInteger(resolved)||resolved<0||resolved>0xffffffff)throw new RangeError('seed must be a uint32');
  let state=resolved,spare;
  const uniform=()=>{
    state=(state+0x6D2B79F5)>>>0;let t=state;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);
    return ((t^(t>>>14))>>>0)/4294967296;
  };
  return {seed:resolved,uniform(count){return Float32Array.from({length:count},uniform);},normal(count){return Float32Array.from({length:count},()=>{
    if(spare!==undefined){const x=spare;spare=undefined;return x;}
    const radius=Math.sqrt(-2*Math.log(1-uniform())),angle=2*Math.PI*uniform();
    spare=radius*Math.sin(angle);return radius*Math.cos(angle);
  });}};
}
export function sequenceEntropy(logits,channels=32) {
  if(logits.length%channels)throw new Error('Invalid sequence logits');
  const out=new Float32Array(logits.length/channels),prob=new Float32Array(channels);
  for(let i=0;i<out.length;i++){
    let maximum=-Infinity;for(let c=0;c<channels;c++)maximum=Math.max(maximum,logits[i*channels+c]);
    let sum=0;for(let c=0;c<channels;c++){prob[c]=Math.exp(logits[i*channels+c]-maximum);sum=f32(sum+prob[c]);}
    let entropy=0;for(let c=0;c<channels;c++){const p=f32(prob[c]/sum);entropy=f32(entropy+f32(p*f32(Math.log(f32(p+1e-10)))));}
    out[i]=-entropy;
  }
  return out;
}

// AF3 Euler update for the upstream default allow_realignment=false path. Shared
// draws and an explicit schedule make the arithmetic reproducible across engines.
function* sampleSteps(engine,{features,coordinates,random,seed,schedule,steps=200,partialT,n_recycle,
  gamma0=0.6,gammaMin=1,noiseScale=1.003,stepScale=1.5,cfgScale=1,referenceFeatures,
  allowRealignment=false,symmetry,signal,onStep,centerOption='all',sTrans=1,retainTrajectory=true,symStepFrac=0.9}={}) {
  const symmetric=Boolean(symmetry||features.sym_transform);
  if(symmetric)validateSymmetry(features,{cfgScale,gamma0,symStepFrac});
  if(![gamma0,gammaMin,noiseScale,stepScale,cfgScale].every(Number.isFinite)||gamma0<0||noiseScale<0||stepScale<=0)
    throw new Error('Invalid sampler parameters');
  const rng=resolveRandom(seed),resolvedSchedule=schedule?Float32Array.from(schedule):noiseSchedule({steps,partialT});
  if(resolvedSchedule.length<2||!resolvedSchedule.every((v,i)=>v>0&&Number.isFinite(v)&&(i===0||v<resolvedSchedule[i-1])))throw new Error('Expected at least two descending positive schedule points');
  if(schedule&&partialT!==undefined)throw new Error('Supply either schedule or partialT');
  if(partialT!==undefined&&!coordinates)throw new Error('Partial diffusion requires initial coordinates');
  const l=features.atom_to_token_map.length,size=l*3,base=coordinates??features.motif_pos;
  if(base.length!==size||!base.every(Number.isFinite))throw new Error('Invalid initial coordinates');
  const fixed=features.is_motif_atom_with_fixed_coord;
  const draw=(value)=>{const a=value??rng.normal(size);if(a.length!==size||!a.every(Number.isFinite))throw new Error('Invalid shared random draw');return a;};
  if(random&&(!random.initial||random.steps?.length!==resolvedSchedule.length-1))throw new Error('Shared randomness must contain all draws');
  if(random&&allowRealignment&&(random.augmentations?.length!==resolvedSchedule.length-1||fixed.some(Boolean)&&!random.finalAugmentation))throw new Error('Shared realignment randomness needs augmentation angles and translations for every step and final motif alignment');
  const initial=draw(random?.initial);
  let x=Float32Array.from(base,(v,i)=>v+(fixed[Math.floor(i/3)]?0:f32(initial[i]*resolvedSchedule[0])));
  let initializer,refInit;
  try {
  initializer=yield ()=>engine.initialize(features);
  if(cfgScale!==1&&!referenceFeatures)throw new Error('CFG requires explicit referenceFeatures');
  refInit=cfgScale!==1?yield ()=>engine.initialize(referenceFeatures):undefined;
  const trajectory=[],entropy=[],times=[];let out;
  for(let step=0;step<resolvedSchedule.length-1;step++) {
    signal?.throwIfAborted();
    if(allowRealignment){const augmentation=random?.augmentations?.[step]??{angles:rng.uniform(3).map(v=>v*2*Math.PI),translation:rng.normal(3)};x=rigidAugment(x,base,fixed,{...augmentation,centerOption,sTrans}).coordinates;}
    const previous=resolvedSchedule[step],next=resolvedSchedule[step+1],gamma=next>gammaMin?gamma0:0,hat=f32(previous*f32(1+gamma));
    const scale=f32(noiseScale*f32(Math.sqrt(f32(f32(hat*hat)-f32(previous*previous))))),noise=draw(random?.steps[step]);
    const noisy=Float32Array.from(x,(v,i)=>v+(fixed[Math.floor(i/3)]?0:f32(scale*noise[i])));
    out=yield ()=>engine.run({features,X_noisy_L:noisy,t:hat,n_recycle,initializer});
    if(symmetric&&next>resolvedSchedule[Math.min(Math.floor(resolvedSchedule.length*symStepFrac),resolvedSchedule.length-1)])out.X_L.data=applySymmetry(out.X_L.data,features,{partialDiffusion:partialT!==undefined||!allowRealignment&&fixed.some((v,i)=>v&&features.is_sym_asu[i]&&features.sym_entity_id[i]!==-1)});
    const denoised=out.X_L.data;
    let delta=Float32Array.from(noisy,(v,i)=>f32(v-denoised[i])/hat);
    if(cfgScale!==1){
      const refSize=referenceFeatures.atom_to_token_map.length*3;
      if(refSize>size)throw new Error('Reference atom count exceeds conditional atom count');
      const refOut=yield ()=>engine.run({features:referenceFeatures,X_noisy_L:noisy.slice(0,refSize),t:hat,n_recycle,initializer:refInit});
      const ref=refOut.X_L.data;
      delta=Float32Array.from(delta,(v,i)=>v+f32(f32(cfgScale-1)*f32(v-(i<refSize?f32(f32(noisy[i]-ref[i])/hat):0))));
    }
    const multiplier=f32(stepScale*f32(next-hat));
    x=Float32Array.from(noisy,(v,i)=>v+f32(multiplier*delta[i]));
    if(retainTrajectory)trajectory.push({noisy,denoised:denoised.slice(),coordinates:x.slice()});times.push(hat);
    entropy.push(sequenceEntropy(out.sequence_logits_I.data));
    onStep?.({step,total:resolvedSchedule.length-1,coordinates:x.slice()});
  }
  if(allowRealignment&&fixed.some(Boolean)){const augmentation=random?.finalAugmentation??{angles:rng.uniform(3).map(v=>v*2*Math.PI),translation:rng.normal(3)};x=rigidAugment(x,base,fixed,{...augmentation}).coordinates;if(symmetric)x=applySymmetry(x,features,{partialDiffusion:partialT!==undefined});x=rigidAlign(base,x,{mask:fixed}).coordinates;}
  return {...out,X_L:{shape:[1,l,3],data:x},trajectory,sequence_entropy_traj:entropy,t_hats:new Float32Array(times),seed:rng.seed,schedule:resolvedSchedule};
  } finally {if(initializer)engine.releaseInitializer?.(initializer);if(refInit)engine.releaseInitializer?.(refInit);}
}
export function sampleCPU(engine,options){const iterator=sampleSteps(engine,options);let next=iterator.next();try{while(!next.done)next=iterator.next(next.value());return next.value;}finally{iterator.return();}}
export async function sampleGPU(engine,options){const iterator=sampleSteps(engine,options);let next=iterator.next();try{while(!next.done)next=iterator.next(await next.value());return next.value;}finally{iterator.return();}}

