import {loadWeights} from '../shared/weights.js';
import {ESM3Engine} from '../shared/api.js';
import {GPUBackend,requestWebGpuDevice} from './backend.js';
export {requestWebGpuDevice};
// Keep an in-flight load rooted until asynchronous browser I/O has settled.
const activeLoads=new Set();
export async function createESM3Engine({baseUrl,precision='int8',device,useF16=false,cache='cache',signal,onProgress,read}={}){
 if(!baseUrl||!['fp32','fp16','int8'].includes(precision))throw new Error('Supply baseUrl and fp32, fp16, or int8 precision');
 const ownsDevice=!device;if(!device)({device}=await requestWebGpuDevice({useF16}));
 const b=new GPUBackend(device,{ownsDevice,useF16});activeLoads.add(b);
 try{b.loadTask=loadWeights(`${baseUrl.replace(/\/$/,'')}/${precision}`,{cache,signal,onProgress,read,onTensor:(k,t)=>b.upload(k,t)});const bundle=await b.loadTask;const e=new ESM3Engine(b,{gpu:true,precision});e.manifest=bundle.manifest;return e;}catch(e){b.destroy();throw e;}finally{delete b.loadTask;activeLoads.delete(b);}
}
