import {loadWeights} from '../shared/weights.js';
import {ESM3Engine} from '../shared/api.js';
import {CPUBackend} from './backend.js';
export async function createESM3CPU({baseUrl,precision='fp32',...options}={}){
 if(!baseUrl||!['fp32','fp16','int8'].includes(precision))throw new Error('Supply baseUrl and fp32, fp16, or int8 precision');
 const b=new CPUBackend();try{const bundle=await loadWeights(`${baseUrl.replace(/\/$/,'')}/${precision}`,{...options,onTensor:(k,t)=>b.upload(k,t)});const e=new ESM3Engine(b,{precision});e.manifest=bundle.manifest;return e;}catch(e){b.destroy();throw e;}
}
