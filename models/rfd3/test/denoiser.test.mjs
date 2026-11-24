import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createRFD3CPUEngine } from '../cpu/index.js';
import { metrics } from '../shared/metrics.js';
test('CPU full denoiser, 18 token blocks and two recycling passes, official checkpoint',async()=>{
  const architecture=JSON.parse(await readFile(new URL('../fixtures/architecture.json',import.meta.url)));
  const baseURL=new URL('../weights/fp32/',import.meta.url);
  const manifest=JSON.parse(await readFile(new URL('manifest.json',baseURL)));
  const engine=await createRFD3CPUEngine({manifest,architecture,baseURL,cache:'none',fetcher:async url=>new Response(await readFile(new URL(url)))});
  const results=[],outputs={};
  try{
    for(const name of ['monomer','cyclic','atomic-motif','ligand','dna','binder-hotspot','unindexed','hbond-exposure','multichain','fixed-backbone','sequence-redesign','partial','rna','atomic-triad']){
      const f=JSON.parse(await readFile(new URL('../fixtures/'+(['monomer','cyclic','atomic-motif'].includes(name)?'oracle':'oracle-pass2')+'/'+name+'.json',import.meta.url)));
      const features=Object.fromEntries(Object.entries(f.features).map(([k,v])=>[k,new Float32Array(v.data)]));
      const start=performance.now();
      const output=engine.run({features,X_noisy_L:new Float32Array(f.X_noisy_L.data),t:f.t.data,n_recycle:2});
      const elapsed=performance.now()-start;outputs[name]=Object.fromEntries(Object.entries(output).map(([k,v])=>[k,{shape:v.shape,data:Array.from(v.data)}]));
      if(name==='atomic-motif'){
        const fixed=features.is_motif_atom_with_fixed_coord;
        assert.equal(fixed.reduce((a,b)=>a+b,0),2);
        assert.ok(features.active_donor.some(x=>x!==0));
        assert.ok(features.ref_atomwise_rasa.some(x=>x!==0));
        for(let i=0;i<fixed.length;i++)if(fixed[i])for(let c=0;c<3;c++)assert.equal(output.X_L.data[i*3+c],f.X_noisy_L.data[i*3+c]);
      }
      if(name==='monomer' && f.trajectory){
        const oracle=f.trajectory;
        const start=performance.now();
        const sampled=engine.sample({features,coordinates:new Float32Array(oracle.coordinates.data),
          schedule:oracle.schedule.data,n_recycle:2,seed:20261123,
          random:{initial:new Float32Array(oracle.random[0].data),steps:oracle.random.slice(1).map(v=>new Float32Array(v.data))}});
        const final=metrics(sampled.X_L.data,oracle.output.X_L.data);
        const entry={case:name,tensor:'trajectory_final',...final,cpuTrajectoryMs:performance.now()-start};
        results.push(entry);
        assert.ok(final.maxAbsDiff<0.01 && final.cosineSim>0.9999999,JSON.stringify(entry));
        for(let i=0;i<sampled.trajectory.length;i++){
          const stats=metrics(sampled.trajectory[i].denoised,oracle.output.X_denoised_L_traj[i].data);
          results.push({case:name,tensor:'trajectory_denoised_'+i,...stats});
          assert.ok(stats.maxAbsDiff<0.01,JSON.stringify(results.at(-1)));
        }
        assert.equal(sampled.seed,20261123);
      }
      for(const [key,v]of Object.entries(output)){
        const stats=metrics(v.data,f.output[key].data);results.push({case:name,tensor:key,...stats,cpuStepMs:elapsed});
        assert.ok(stats.maxAbsDiff < (key === 'sequence_logits_I' ? 0.1 : 0.001) && stats.cosineSim > 0.99999999,JSON.stringify(results.at(-1)));
      }
    }
  }finally{engine.destroy();}
  await writeFile(new URL('../fixtures/cpu-output.json',import.meta.url),JSON.stringify(outputs)+'\n');
  await writeFile(new URL('../fixtures/cpu-denoiser-parity.json',import.meta.url),JSON.stringify(results,null,2)+'\n');
  console.log(JSON.stringify({denoiser:results}));
});
