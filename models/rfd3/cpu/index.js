import { validateFeatures } from '../shared/features.js';
import { generate } from '../shared/input.js';
import { iterateWeights } from '../shared/weights.js';
import { RFD3CPUDenoiser } from './denoiser.js';
import { sampleCPU } from '../shared/sampler.js';

// Pure JavaScript reference engine; use a worker for synchronous neural execution.
export async function createRFD3CPUEngine({ manifest, architecture, baseURL, ...options }) {
  if (!architecture?.config || architecture.checkpoint_sha256 !== manifest?.checkpoint_sha256)
    throw new Error('Architecture/checkpoint mismatch');
  const weights=new Map();
  for await(const tensor of iterateWeights(manifest,baseURL,options))
    weights.set(tensor.name,{...tensor,bytes:tensor.bytes.slice(),scales:tensor.scales?.slice()});
  const valid=manifest.constants['diffusion_module.sequence_head.valid_out_mask']?.value;
  if(!Array.isArray(valid)||valid.length!==32)throw new Error('Missing sequence decoding mask');
  let model=new RFD3CPUDenoiser(weights,{...architecture.config,sequence_valid_out_mask:valid});
  function live(){if(!model)throw new Error('CPU engine is destroyed');options.signal?.throwIfAborted();}
  return {
    precision:manifest.precision,
    initialize(features){live();validateFeatures(features,architecture.config);return model.initialize(features);},
    run({features,X_noisy_L,t,n_recycle,initializer,random,...sampling}){
      if(X_noisy_L===undefined&&random)return this.sample({features,n_recycle,random,...sampling});
      live();validateFeatures(features,architecture.config);
      if(!(X_noisy_L instanceof Float32Array)||!X_noisy_L.every(Number.isFinite))throw new Error('run requires explicit Float32Array noisy coordinates');
      const atoms=features.atom_to_token_map.length,batch=X_noisy_L.length/(atoms*3);
      if(!Number.isInteger(batch)||batch<1)throw new Error('Invalid coordinate dimensions');
      const times=typeof t==='number'?Array(batch).fill(t):t;
      if(times?.length!==batch||!Array.from(times).every(t=>Number.isFinite(t)&&t>=0))throw new Error('Expected one noise time per sample');
      const init=initializer??model.initialize(features),samples=[];
      for(let b=0;b<batch;b++)samples.push(model.denoise({features,X_noisy_L:X_noisy_L.subarray(b*atoms*3,(b+1)*atoms*3),t:times[b],n_recycle,initializer:init}));
      const output={};
      for(const key of Object.keys(samples[0])){
        const first=samples[0][key],data=new first.data.constructor(first.data.length*batch);
        samples.forEach((sample,i)=>data.set(sample[key].data,i*first.data.length));
        output[key]={shape:[batch,...first.shape.slice(1)],data};
      }
      return output;
    },
    generate(options){return generate(this,options);},
    sample(options){live();return sampleCPU(this,options);},
    destroy(){weights.clear();model=null;},
  };
}
