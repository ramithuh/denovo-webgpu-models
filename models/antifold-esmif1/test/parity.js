import {diffStats,prepareInput} from '../shared/index.js';
export function oracleTensor(meta,bytes,name){const t=meta.tensors[name];if(!t)throw new Error(`Missing oracle tensor ${name}`);return new (t.dtype==='I32'?Int32Array:Float32Array)(bytes,t.offset,t.length);}
export function compareOutput(result,c,meta,bytes){
 const p=prepareInput(c),actual={...result.intermediates,coords:p.coords,padding:p.padding,coordMask:p.coordMask,confidence:p.confidence,tokens:Int32Array.from([33,...p.tokens]),logProbabilities:result.logProbabilities,canonicalProbabilities:result.canonicalProbabilities,entropyPerplexity:result.entropyPerplexity},metrics={};
 for(const [name,t] of Object.entries(meta.tensors)){
  if(name.startsWith('replay.')||name.startsWith('independent.'))continue;
  let a=actual[name],b=oracleTensor(meta,bytes,name);if(!a)throw new Error(`Missing runtime trace ${name}`);
  if(name.startsWith('edge')||name.startsWith('graph.edge')){
   const indices=actual.edgeIndex,original=oracleTensor(meta,bytes,'edgeIndex'),lookup=new Map();for(let i=0;i<indices.length/2;i++)if(indices[2*i]>=0)lookup.set(`${indices[2*i]},${indices[2*i+1]}`,i);
   if(lookup.size!==original.length/2)throw new Error('Graph edge count mismatch');
   const d=a.length/(indices.length/2),aligned=new Float32Array(b.length);for(let i=0;i<original.length/2;i++){const j=lookup.get(`${original[2*i]},${original[2*i+1]}`);if(j===undefined)throw new Error('Graph neighbor mismatch');aligned.set(a.subarray(j*d,(j+1)*d),i*d);}a=aligned;
  }
  if(name==='coords'){if(a.length!==b.length)throw new Error('Input shape mismatch');for(let i=0;i<a.length;i++)if(!Object.is(a[i],b[i]))throw new Error(`Coordinates differ at ${i}`);metrics[name]={exact:true};continue;}
  const stats=diffStats(a,b);metrics[name]=stats;if(t.dtype==='I32'&&stats.maxAbs!==0)throw new Error(`Non-exact tokens/graph/mask ${name}`);if(!(stats.relRMS<1e-4))throw new Error(`${c.name}/${meta.precision}/${name}: relRMS ${stats.relRMS} >= 1e-4`);
 }
 const scores=Float32Array.from([result.logLikelihood,result.perplexity,result.coordinateLogLikelihood,result.coordinatePerplexity]);if(!meta.scores)throw new Error("Missing reference aggregate scores");metrics.scores=diffStats(scores,meta.scores);if(!(metrics.scores.relRMS<1e-4))throw new Error(`Aggregate likelihood/perplexity mismatch ${metrics.scores.relRMS}`);
 return metrics;
}
export function checkReplay(result,meta,bytes){const tokens=oracleTensor(meta,bytes,'replay.tokens'),logits=oracleTensor(meta,bytes,'replay.logits');if(result.tokens.toString()!==tokens.toString())throw new Error('Autoregressive replay tokens differ');const stats=diffStats(result.logits,logits);if(!(stats.relRMS<1e-4))throw new Error(`Replay arithmetic mismatch ${stats.relRMS}`);return stats;}
