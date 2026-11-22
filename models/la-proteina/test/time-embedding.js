import {LaProteinaNetwork} from '../shared/network.js';
import {fixture,compare} from './parity.js';
import {center} from '../shared/sampling.js';

export async function centeringTest(ops,manifest,read){
  const ref=await fixture(manifest,read),results={};
  try{
    for(const name of Object.keys(ref).filter(n=>n.startsWith('x'))){
      const length=Number(name.slice(1));
      results[length]=compare(await ops.read(center(ops,ops.tensor([length,3],ref[name]))),ref['y'+length],'centering '+length,1e-7);
      await ops.checkpoint([]);
    }return results;
  }finally{ops.clear();}
}

// Large angles catch loss of precision in native GPU sine/cosine intrinsics.
export async function timeEmbeddingTest(ops,manifest,read){
  const ref=await fixture(manifest,read),network=new LaProteinaNetwork(ops,{}, {t_emb_dim:256},'LD1');
  const result=new Float32Array(ref.embedding.length);
  try{
    for(let i=0;i<ref.times.length;i++){
      result.set((await ops.read(network.time(ref.times[i],ref.times[i]))).subarray(0,256),i*256);
      await ops.checkpoint([]);
    }
    return compare(result,ref.embedding,'time embeddings',1e-6);
  }finally{ops.clear();}
}
