import { loadBundle, toFloat, abort } from './loader.js';
import { LaProteinaNetwork } from './network.js';
import { sample, validateLength, validateArray } from './sampling.js';
import { structure } from './pdb.js';

export async function makeEngine(ops,options={}) {
  const {baseUrl,precision='fp32',cache='none',read,signal,onProgress}=options;
  if(!baseUrl)throw Error('baseUrl is required');
  if(!['fp32','fp16','int8'].includes(precision))throw Error('Unsupported precision');
  let destroyed=false,busy=false;const models={};const manifests={};
  try{
    for(const name of ['LD1','AE1']){
      const bundle=await loadBundle({baseUrl:`${String(baseUrl).replace(/\/$/,'')}/${name.toLowerCase()}/${precision}`,cache,read,signal,onProgress});
      if(bundle.manifest.model!==name||bundle.manifest.precision!==precision)throw Error('Manifest model or precision mismatch');
      const weights={};
      for(const [k,v] of Object.entries(bundle.weights)){
        abort(signal);
        weights[k]=ops.uploadWeight?ops.uploadWeight(v):ops.upload(v.shape,toFloat(v),true);
        delete bundle.weights[k];
      }
      manifests[name]=bundle.manifest;
      models[name]=new LaProteinaNetwork(ops,weights,bundle.manifest.config,name);
    }
  }catch(e){ops.destroy();throw e;}
  async function run(fn,opts={}) {
    if(destroyed)throw Error('Engine destroyed');
    if(busy)throw Error('Concurrent inference on one engine is unsupported');
    busy=true;const start=performance.now();
    try{abort(opts.signal);const result=await fn();abort(opts.signal);return {...result,timingMs:performance.now()-start};}
    finally{ops.clear();busy=false;}
  }
  return {
    precision,manifests,
    sample(opts={}) {return run(async()=>structure(await sample(ops,models.LD1,models.AE1,opts)),opts);},
    denoiseStep(input,opts={}) {return run(async()=>{
      const L=input.length;validateLength(L);
      const x={ca:ops.tensor([L,3],validateArray(input.ca,L*3,'ca')),z:ops.tensor([L,8],validateArray(input.z,L*8,'z')),tCa:input.tCa,tZ:input.tZ};
      for(const t of [x.tCa,x.tZ])if(!Number.isFinite(t)||t<0||t>=1)throw Error('Time must be in [0, 1)');
      if(input.scCa)x.scCa=ops.tensor([L,3],validateArray(input.scCa,L*3,'scCa'));
      if(input.scZ)x.scZ=ops.tensor([L,8],validateArray(input.scZ,L*8,'scZ'));
      const trace=opts.returnIntermediates?{}:undefined;
      const out=await models.LD1.forward(x,{...opts,trace});
      return {ca:await ops.read(out.ca),z:await ops.read(out.z),trace};
    },opts);},
    decode(input,opts={}) {return run(async()=>{
      const L=input.length;validateLength(L);
      const ca=ops.tensor([L,3],validateArray(input.ca,L*3,'ca')),z=ops.tensor([L,8],validateArray(input.z,L*8,'z'));
      const trace=opts.returnIntermediates?{}:undefined;
      const out=await models.AE1.forward({ca,z},{...opts,trace});
      return structure({coords:await ops.read(out.coords),logits:await ops.read(out.logits),ca:await ops.read(ca),z:await ops.read(z),trace});
    },opts);},
    memoryUsage:()=>ops.memoryUsage(),
    destroy(){if(destroyed)return;destroyed=true;ops.destroy();},
  };
}
