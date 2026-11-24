import {readFile,writeFile}from'node:fs/promises';
import {buildInput}from'../shared/input.js';
import {metrics}from'../shared/metrics.js';
const root=new URL('../',import.meta.url),read=async p=>JSON.parse(await readFile(new URL(p,root)));
const [cpu,gpu,geometry]=await Promise.all([read('fixtures/cpu-denoiser-parity.json'),read('fixtures/gpu-model-parity.json'),read('fixtures/generation-geometry.json')]);
const number=n=>n===0?'0':n.toExponential(6),lines=[];
const featureParity=[];
for(const name of new Set(gpu.model.filter(r=>r.precision==='fp32'&&r.stage==='denoiser').map(r=>r.case))){
 const directory=['monomer','cyclic','atomic-motif'].includes(name)?'oracle':'oracle-pass2';
 const oracle=await read('fixtures/'+directory+'/'+name+'.json'),spec={...oracle.specification};
 if(spec.input)spec.input=await readFile(new URL(spec.input,root),'utf8');
 spec.cyclic=Boolean(spec.cyclic_chains);
 const {features}=buildInput({...spec,cleanupGuideposts:false});
 for(const [tensor,value]of Object.entries(oracle.features))featureParity.push({case:name,tensor,...metrics(features[tensor],Float32Array.from(value.data))});
}
await writeFile(new URL('fixtures/featurizer-parity.json',root),JSON.stringify(featureParity,null,2)+'\n');
lines.push(`Across ${new Set(featureParity.map(r=>r.case)).size} featurizer cases, the worst tensor maxAbsDiff is **${number(Math.max(...featureParity.map(r=>r.maxAbsDiff)))}**, worst RMSE is **${number(Math.max(...featureParity.map(r=>r.rmse)))}**, and minimum cosineSim is **${Math.min(...featureParity.map(r=>r.cosineSim)).toFixed(14)}**. [Per-tensor feature metrics](fixtures/featurizer-parity.json).`,'');
lines.push('| GB10 single denoiser vs PyTorch FP32 (worst across conditioned cases) | maxAbsDiff Å | max RMSE Å | min cosineSim |','|---|---:|---:|---:|');
for(const precision of ['fp32','fp16','int8']){const rows=gpu.model.filter(r=>r.precision===precision&&r.stage==='denoiser'&&r.tensor==='X_L');lines.push(`| ${precision} | ${number(Math.max(...rows.map(r=>r.maxAbsDiff)))} | ${number(Math.max(...rows.map(r=>r.rmse)))} | ${Math.min(...rows.map(r=>r.cosineSim)).toFixed(14)} |`);}
lines.push('','| FP32 conditioned denoiser | GPU maxAbsDiff Å | GPU RMSE Å | GPU cosineSim | CPU maxAbsDiff Å |','|---|---:|---:|---:|---:|');
for(const row of gpu.model.filter(r=>r.precision==='fp32'&&r.stage==='denoiser'&&r.tensor==='X_L')){const reference=cpu.find(r=>r.case===row.case&&r.tensor==='X_L');lines.push(`| ${row.case} | ${number(row.maxAbsDiff)} | ${number(row.rmse)} | ${row.cosineSim.toFixed(14)} | ${reference?number(reference.maxAbsDiff):'—'} |`);}
const cpuReference=gpu.model.filter(r=>r.stage==='cpu-reference'&&r.tensor==='X_L');
lines.push('',`GPU vs JavaScript CPU FP32 coordinates: worst maxAbsDiff **${number(Math.max(...cpuReference.map(r=>r.maxAbsDiff)))} Å**, worst RMSE **${number(Math.max(...cpuReference.map(r=>r.rmse)))} Å**, minimum cosineSim **${Math.min(...cpuReference.map(r=>r.cosineSim)).toFixed(14)}**.`);
const initial=gpu.model.filter(r=>r.precision==='fp32'&&r.stage==='initializer'),logits=gpu.model.filter(r=>r.precision==='fp32'&&r.stage==='denoiser'&&r.tensor==='sequence_logits_I');
lines.push('',`FP32 GPU initializer worst maxAbsDiff: **${number(Math.max(...initial.map(r=>r.maxAbsDiff)))}**; sequence logits: **${number(Math.max(...logits.map(r=>r.maxAbsDiff)))}**. CPU sequence-logit worst maxAbsDiff: **${number(Math.max(...cpu.filter(r=>r.tensor==='sequence_logits_I').map(r=>r.maxAbsDiff)))}**. Raw logits have much larger magnitudes than coordinates. FP32 decoded indices are asserted identical to PyTorch.`,
'','| Packed full model vs PyTorch with the same quantized weights | Worst coordinate maxAbsDiff Å | Worst logit maxAbsDiff |','|---|---:|---:|');
for(const precision of ['fp16','int8']){const rows=gpu.model.filter(r=>r.precision===precision&&r.stage==='quantized-denoiser');lines.push(`| ${precision} | ${number(Math.max(...rows.filter(r=>r.tensor==='X_L').map(r=>r.maxAbsDiff)))} | ${number(Math.max(...rows.filter(r=>r.tensor==='sequence_logits_I').map(r=>r.maxAbsDiff)))} |`);}
lines.push('','| Shared-draw trajectory vs PyTorch FP32 (four schedule points) | Precision | maxAbsDiff Å | RMSE Å | cosineSim |','|---|---|---:|---:|---:|');
for(const row of gpu.model.filter(r=>r.stage==='trajectory'&&r.tensor==='X_L'))lines.push(`| ${row.case} | ${row.precision} | ${number(row.maxAbsDiff)} | ${number(row.rmse)} | ${row.cosineSim.toFixed(14)} |`);
lines.push('','These deliberately short, high-noise trajectories test arithmetic, not protein quality. Quantized trajectories are not expected to match FP32 coordinates exactly. Real generation below uses 20–50 updates.',
'','| GB10 generation | Precision | Tokens / atoms | Updates | Warm step mean ms | Trajectory s | Peak GPU buffers GB |','|---|---|---:|---:|---:|---:|---:|');
for(const r of geometry)lines.push(`| ${r.case} | ${r.precision} | ${r.tokens} / ${r.atoms} | ${r.updates} | ${r.warmStepMeanMs.toFixed(2)} | ${r.wallSeconds.toFixed(3)} | ${(r.peakBytes/1e9).toFixed(3)} |`);
lines.push('','| Geometry (seed 20261123) | Precision | Protein residues | Breaks | Cα mean Å | Severe overlaps | vdW overlaps | Motif RMSD Å | PSEA helix / strand / coil |','|---|---|---:|---:|---:|---:|---:|---:|---|');
for(const r of geometry){const s=r.secondaryStructurePSEA;lines.push(`| ${r.case} | ${r.precision} | ${r.proteinResidues} | ${r.chainBreaks} | ${r.caMean?.toFixed(4)??'—'} | ${r.severeClashes} | ${r.vdwOverlapPairs} | ${r.motifRmsd==null?'—':number(r.motifRmsd)} | ${s?`${s.helix} / ${s.strand} / ${s.coil}`:'—'} |`);}
const cyclic=geometry.find(r=>r.cyclicClosure),unindexed=geometry.find(r=>r.case==='unindexed');
if(cyclic)lines.push('',`Cyclic closing C–N: **${cyclic.cyclicClosure.CN.toFixed(4)} Å**; closing Cα–Cα: **${cyclic.cyclicClosure.CA.toFixed(4)} Å**.`);
if(unindexed)lines.push(`Unindexed pre-insertion motif RMSD: **${unindexed.motifAssignments[0].insertionRmsd.toFixed(6)} Å**. Export cleanup inserts the fixed atoms; the largest resulting peptide C–N distance is **${unindexed.peptideCNMax.toFixed(4)} Å**. This is reported separately from raw guidepost preservation.`);
const path=new URL('README_WEBGPU.md',root),document=await readFile(path,'utf8');await writeFile(path,document.replace(/<!-- MEASUREMENTS -->[\s\S]*?<!-- \/MEASUREMENTS -->/,'<!-- MEASUREMENTS -->\n'+lines.join('\n')+'\n<!-- /MEASUREMENTS -->'));
