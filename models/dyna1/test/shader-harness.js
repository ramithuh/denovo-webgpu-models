import {verifyPackedAttention} from './packed-attention.js';
import * as s from '../webgpu/dyna1/shaders.js';
import {requestWebGpuDevice} from '../webgpu/index.js';
export async function run(){
 const {device,hasF16}=await requestWebGpuDevice();const sources=[];
 for(const dtype of ['F32','F16','I8']) for(const bias of [false,true]) {
  sources.push(s.matmulSource(20,1536,1536,dtype,bias,false));
  if(dtype==='F16'&&hasF16) sources.push(s.matmulSource(20,1536,1536,dtype,bias,true));
 }
 sources.push(
  s.normSource(20,1536),
  s.normSource(20,1536,{stride:4608,offset:1536,bias:false}),
  s.elementSource(20,'y[i]=x[i]+b[i];',2),
  s.gatherSource(20,1536),
  s.gatherAddSource(20,1536),
  s.ropeSource(20,1536,24),
  s.scoresSource(20,1536,24),
  s.softmaxSource(20,24),
  s.contextSource(20,1536,24),
  s.geomRotateSource(320,128),
  s.geomScoresSource(16,128,20),
  s.geomContextSource(16,128,20),
  s.selectLocalSource(20,16,1024),
  s.codebookArgminSource(20,4096,128),
 );
 for(const code of sources){device.pushErrorScope('validation');const module=device.createShaderModule({code}),info=await module.getCompilationInfo();if(info.messages.some(m=>m.type==='error'))throw new Error(info.messages.map(m=>m.message).join('\n'));await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});const error=await device.popErrorScope();if(error)throw new Error(error.message);}
 const packedScoresChecked=await verifyPackedAttention(device);
 device.destroy();return {shadersCompiled:sources.length,hasF16,packedScoresChecked};
}
