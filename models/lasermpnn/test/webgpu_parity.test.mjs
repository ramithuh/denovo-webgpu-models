import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import { browserRun, gpuSetup } from "./browser.mjs";
test("Hardware FP32 holo inference and FP16 arithmetic parity against matching PyTorch precision", async () => {
  const result = await browserRun(`
 const {GpuBackend}=await import('/webgpu/lasermpnn/backend.js');const {LaserModel}=await import('/shared/lasermpnn/model.js');const {loadWeights}=await import('/shared/lasermpnn/weights.js');const {makeGeometry}=await import('/shared/lasermpnn/geometry.js');
 ${gpuSetup}
 const meta=await(await fetch('/fixtures/oracle_lasermpnn.json')).json(),buffer=await(await fetch('/fixtures/oracle_lasermpnn.bin')).arrayBuffer();
 const read=n=>{const r=meta.tensors[n],buf=buffer.slice(r.byteOffset,r.byteOffset+r.byteLength);return r.dtype==='I32'?new Int32Array(buf):new Float32Array(buf);};
 const idx=k=>{const a=read(k),e=a.length/2;return [Array.from(a.slice(0,e)),Array.from(a.slice(e))];};
 const batch={length:meta.length,backbone:read('input.backbone_coords'),ligandCoords:read('ligand.lig_coords'),ligandNodes:read('ligand.lig_nodes'),pp:idx('input.pr_pr_edge_index'),lp:idx('input.lig_pr_edge_index'),ll:idx('ligand.lig_lig_edge_index'),ppDistances:read('input.pr_pr_edge_distance'),lpDistances:read('input.lig_pr_edge_distance'),llDistances:read('ligand.lig_lig_edge_distance'),sequence:read('input.sequence_indices'),chi:read('input.chi_angles'),fixed:read('input.chain_mask'),decodingOrder:Array.from(read('input.decoding_order'))};
 const metrics={};function compare(a,b,name,tolerance=1e-4){if(a.length!==b.length)throw Error(name+' shape mismatch '+a.length+' '+b.length);let error=0,ref=0,maxAbs=0;for(let i=0;i<a.length;i++){if(Number.isNaN(b[i])&&Number.isNaN(a[i]))continue;if(!Number.isFinite(a[i])||!Number.isFinite(b[i]))throw Error(name+' unexpected nonfinite @'+i);const d=a[i]-b[i];error+=d*d;ref+=b[i]*b[i];maxAbs=Math.max(maxAbs,Math.abs(d));}const relRMS=Math.sqrt(error/Math.max(ref,1e-30));metrics[name]={relRMS,maxAbs};if(relRMS>=tolerance)throw Error(name+' relRMS='+relRMS);}
 const weights=await loadWeights('/weights/fp32',{cache:false});await log('Weights loaded; running full 115-residue GPU inference');
 const uncaptured=[];device.addEventListener('uncapturederror',e=>uncaptured.push(e.error.message));device.pushErrorScope('validation');
 const backend=await GpuBackend.create(device,weights.tensors),model=new LaserModel(backend,weights.manifest.config);const start=performance.now();
 const output=await model.run(batch,{trace:true,uniforms:read('rng.categorical_uniforms'),onProgress:p=>{if(p.completed%25===0)log('Decoded '+p.completed+'/'+p.total);}});const timingMs=performance.now()-start;
 for(const [k,v] of Object.entries(output.traces))compare(v,read(k),k);
 compare(output.sequence,read('output.sampled_sequence_indices'),'sequence');compare(output.chiAngles,read('output.sampled_chi_degrees'),'chiAngles');
 const built=makeGeometry(weights.tensors).build(batch.backbone,output.sequence,output.chiAngles);compare(built.coords,read('output.coords'),'coords');
 
 backend.destroy();
 await log('FP32 passed; testing FP16 core and shader-f16 against identically rounded upstream weights');
 const m16=await(await fetch('/fixtures/oracle_lasermpnn_fp16.json')).json(),b16=await(await fetch('/fixtures/oracle_lasermpnn_fp16.bin')).arrayBuffer();
 const read16=n=>{const r=m16.tensors[n],buf=b16.slice(r.byteOffset,r.byteOffset+r.byteLength);return r.dtype==='I32'?new Int32Array(buf):new Float32Array(buf);};
 const weights16=await loadWeights('/weights/fp16',{cache:false}),fp16Timings={};
 for(const shaderF16 of [false,true]){
  const label=shaderF16?'fp16-shader':'fp16-core',b=await GpuBackend.create(device,weights16.tensors,{shaderF16}),m=new LaserModel(b,weights16.manifest.config),start=performance.now();
  const result=await m.run(batch,{trace:true,uniforms:read('rng.categorical_uniforms'),choices:read('rng.categorical_indices')});fp16Timings[label]=performance.now()-start;
  for(const k of ['sequence_output_layer','chi_prediction_layers.0','chi_prediction_layers.1','chi_prediction_layers.2','chi_prediction_layers.3'])compare(result.traces[k],read16(k),label+'.'+k,5e-4);
  compare(result.chiAngles,read16('output.sampled_chi_degrees'),label+'.chiAngles',5e-4);
  compare(makeGeometry(weights16.tensors).build(batch.backbone,result.sequence,result.chiAngles).coords,read16('output.coords'),label+'.coords',5e-4);
  b.destroy();await log(label+' passed');
 }

 await log('Testing public design API from the real PDB, PDB output and warm latency');
 const {createLaserMpnnEngine}=await import('/webgpu/index.js'),{parseStructure}=await import('/shared/lasermpnn/structure.js'),{constants}=await import('/shared/lasermpnn/constants.js');
 const engine=await createLaserMpnnEngine({device,weights,precision:'fp32'}),pdb=await(await fetch('/example_pdbs/4jnj-1_prot.pdb')).text(),apiTimings=[];
 const expectedSequence=Array.from(read('output.sampled_sequence_indices'),i=>weights.manifest.config.alphabet[i]).join('');
 for(let repeat=0;repeat<3;repeat++){
  const result=await engine.design(pdb,{uniforms:read('rng.categorical_uniforms'),decodingOrder:batch.decodingOrder});apiTimings.push(result.timingMs);
  if(result.sequences[0]!==expectedSequence)throw Error('Public API sequence mismatch');
  compare(result.chiAngles[0],read('output.sampled_chi_degrees'),'api.chiAngles');
  if(!Number.isFinite(result.scores[0].totalLogLikelihood))throw Error('Invalid log likelihood');
  compare(result.scores[0].perResidue.map(x=>x.sequence),read('output.sequence_log_probs'),'api.sequenceLogProb');
  const chiRef=read('output.chi_log_probs');compare(result.scores[0].perResidue.map(x=>x.chi),Array.from({length:meta.length},(_,i)=>chiRef.slice(i*4,i*4+4).reduce((a,b)=>a+b,0)),'api.chiLogProb');
  if(repeat===0){
   const structure=parseStructure(result.pdbTexts[0]),coordinates=read('output.cleaned_coords'),actual=[],expected=[];
   if(structure.ligands.length!==meta.ligand_atoms)throw Error('Output lost ligand atoms');
   structure.residues.forEach((r,i)=>{const seq=read('output.sampled_sequence_indices')[i];constants.atomNames[seq].forEach((name,j)=>{const at=(i*24+j)*3;if(!Number.isFinite(coordinates[at]))return;const atom=r.atoms.get(name);if(!atom)throw Error('Missing output atom '+name);actual.push(...atom.xyz);expected.push(...coordinates.slice(at,at+3));});});
   compare(actual,expected,'api.pdbCoordinates');
  }
 }
 engine.destroy();engine.destroy();let rejected=false;try{await engine.design(pdb);}catch{rejected=true;}if(!rejected)throw Error('destroy must invalidate engine');
 const probe=device.createBuffer({size:4,usage:GPUBufferUsage.COPY_DST});probe.destroy();
 await device.queue.onSubmittedWorkDone();const validation=await device.popErrorScope();if(validation||uncaptured.length)throw Error(validation?.message??uncaptured.join('\\n'));
 const quantizationMetrics=await(await fetch('/fixtures/export_metrics.json')).json();const fp16OriginalFp32LimitMet=['sequence_output_layer',...Array.from({length:4},(_,i)=>'chi_prediction_layers.'+i)].every(k=>quantizationMetrics.fp16.conditional_forward[k].relRMS<5e-4);
 return {adapter:{vendor:info.vendor,architecture:info.architecture,device:info.device,isFallbackAdapter:info.isFallbackAdapter},timingMs,fp16Timings,apiTimings,fp16OriginalFp32LimitMet,metrics};
 `);
  assert.ok(result.ok, result.error);
  await fs.mkdir(new URL("../test-results/", import.meta.url), {
    recursive: true,
  });
  await fs.writeFile(
    new URL("../test-results/webgpu.json", import.meta.url),
    JSON.stringify(result, null, 2),
  );
  console.log(JSON.stringify(result, null, 2));
});
