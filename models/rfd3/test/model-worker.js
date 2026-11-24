import { createRFD3Engine } from '../webgpu/index.js';
import { metrics } from '../shared/metrics.js';
async function run(){
 const neighbourOracle=await(await fetch('/fixtures/neighbour-oracle.json')).json();
 const samplerOracle=await(await fetch('/fixtures/sampler-oracle.json')).json();
 const quantized=await(await fetch('/fixtures/quantized-oracle.json')).json();
 const cpu=await(await fetch('/fixtures/cpu-output.json')).json();
 const architecture=await(await fetch('/fixtures/architecture.json')).json(),results=[];
 for(const precision of ['fp32','fp16','int8']){
  const baseURL=new URL('/weights/'+precision+'/',location.href),manifest=await(await fetch(new URL('manifest.json',baseURL))).json();
  const engine=await createRFD3Engine({manifest,architecture,baseURL,cache:'none'});
  try{
   for(const name of ['monomer','cyclic','atomic-motif','ligand','dna','binder-hotspot','unindexed','hbond-exposure','multichain','fixed-backbone','sequence-redesign','partial','rna','atomic-triad']){
    const fixture=await(await fetch('/fixtures/'+(['monomer','cyclic','atomic-motif'].includes(name)?'oracle':'oracle-pass2')+'/'+name+'.json')).json();
    const features=Object.fromEntries(Object.entries(fixture.features).map(([k,v])=>[k,new Float32Array(v.data)]));
    const handle=await engine.initialize(features),initialized=await engine.readInitializer(handle);
    if(name==='multichain'&&precision==='fp32')for(const token of [false,true]){const oracle=neighbourOracle[token?'token':'atom'],actual=await engine.inspectNeighbours(handle,new Float32Array(oracle.coordinates),{token}),length=oracle.tokens.length,expected=new Float32Array(length*length);for(let i=0;i<length;i++)for(const j of oracle.indices[0][i])expected[i*length+j]++;const differences=Array.from(actual).flatMap((v,i)=>v!==expected[i]?[{row:Math.floor(i/length),col:i%length,actual:v,expected:expected[i]}]:[]);results.push({precision,case:name,stage:'neighbours',tensor:token?'token':'atom',differenceCount:differences.length,differences:differences.slice(0,50)});}
    for(const [key,value]of Object.entries(initialized))results.push({precision,case:name,stage:'initializer',tensor:key,...metrics(value.data,fixture.initializer[key].data)});
    const start=performance.now(),out=await engine.run({features,X_noisy_L:new Float32Array(fixture.X_noisy_L.data),t:fixture.t.data,n_recycle:2,initializer:handle}),coldMs=performance.now()-start;
    const warm=performance.now();await engine.run({features,X_noisy_L:new Float32Array(fixture.X_noisy_L.data),t:fixture.t.data,n_recycle:2,initializer:handle});const warmMs=performance.now()-warm;
    for(const [key,value]of Object.entries(out))results.push({precision,case:name,stage:'denoiser',tensor:key,...metrics(value.data,fixture.output[key].data),coldMs,warmMs,peakBytes:engine.peakMemoryUsage()});
    if(precision==='fp32'&&cpu[name])for(const [key,value]of Object.entries(out))results.push({precision,case:name,stage:'cpu-reference',tensor:key,...metrics(value.data,cpu[name][key].data)});
    if(quantized[precision]?.[name])for(const [key,value]of Object.entries(out))results.push({precision,case:name,stage:'quantized-denoiser',tensor:key,...metrics(value.data,quantized[precision][name][key].data)});
    if(name==='monomer')for(const recycle of [1,3]){const value=await engine.run({features,X_noisy_L:new Float32Array(fixture.X_noisy_L.data),t:fixture.t.data,n_recycle:recycle,initializer:handle});results.push({precision,case:name,stage:'recycles-'+recycle,tensor:'X_L',...metrics(value.X_L.data,samplerOracle.recycles[recycle].X_L.data)});}
    engine.releaseInitializer(handle);
    if(fixture.trajectory){const tr=fixture.trajectory;const start=performance.now();const sampled=await engine.sample({features,coordinates:new Float32Array(tr.coordinates.data),schedule:tr.schedule.data,random:{initial:new Float32Array(tr.random[0].data),steps:tr.random.slice(1).map(v=>new Float32Array(v.data))},n_recycle:2,seed:0});results.push({precision,case:name,stage:'trajectory',tensor:'X_L',...metrics(sampled.X_L.data,tr.output.X_L.data),wallMs:performance.now()-start,peakBytes:engine.peakMemoryUsage()});}

   }
   for(const name of ['aligned-motif','symmetry']){
    const fixture=samplerOracle[name],features=Object.fromEntries(Object.entries(fixture.features).map(([k,v])=>[k,k==='sym_transform'?v:new Float32Array(v.data)])),normals=fixture.normal,uniform=fixture.uniform;
    const random={initial:new Float32Array(normals[0].data),steps:normals.slice(1).filter(v=>v.shape[1]!==1).map(v=>new Float32Array(v.data))};
    if(name==='aligned-motif'){const translations=normals.filter(v=>v.shape[1]===1),aug=translations.map((v,i)=>({translation:v.data,angles:[0,1,2].map(c=>Math.fround(uniform[i*3+c].data[0]*Math.fround(2*Math.PI)))}));random.finalAugmentation=aug.pop();random.augmentations=aug;}
    const out=await engine.sample({features,coordinates:new Float32Array(fixture.coordinates.data),schedule:fixture.schedule.data,random,seed:0,n_recycle:2,allowRealignment:name==='aligned-motif'});results.push({precision,case:name,stage:'trajectory',tensor:'X_L',...metrics(out.X_L.data,fixture.output.X_L.data)});
   }
  }finally{engine.destroy();if(engine.memoryUsage()!==0)throw new Error('GPU buffer accounting leaked after model destruction');}
 }
 return {model:results};
}
run().then(postMessage,error=>postMessage({error:error.stack}));
