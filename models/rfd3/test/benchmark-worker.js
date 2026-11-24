import { createRFD3Engine } from '../webgpu/index.js';
import { buildInput } from '../shared/input.js';
async function run(){
 const architecture=await(await fetch('/fixtures/architecture.json')).json(),results=[];
 const structure=await(await fetch('/fixtures/conditioning.pdb')).text(),ligand=await(await fetch('/fixtures/ligand.pdb')).text(),dna=await(await fetch('/fixtures/dna.pdb')).text(),rna=await(await fetch('/fixtures/rna.pdb')).text();
 const cases=[{name:'monomer',mode:'unconditional',length:64,steps:51},{name:'motif',mode:'motif',structure,contig:'30,A1,33',fixedAtoms:{A1:'N,CA,C,O'},steps:51},
 {name:'ligand',mode:'conditioned',structure:ligand,length:24,ligand:'ATP',steps:21},{name:'dna',mode:'conditioned',structure:dna,contig:'24,/0,D1',steps:21},
 {name:'cyclic',length:12,cyclic:true,steps:31},{name:'hbond-exposure',mode:'motif',structure,contig:'12,A1,11',fixedAtoms:{A1:'N,CA'},hbondDonor:{A1:'N'},hbondAcceptor:{A1:'O'},select_buried:{A1:'CA'},steps:21},
 {name:'unindexed',mode:'motif',structure,length:24,unindex:'A1',fixedAtoms:{A1:'N,CA'},steps:21},
 {name:'binder',mode:'binder',structure,length:24,target:'A1',hotspots:{A1:'CA,CB'},steps:21},
 {name:'rna',mode:'conditioned',structure:rna,contig:'24,/0,R1',steps:21},
 {name:'realignment',mode:'motif',structure,contig:'12,A1,11',fixedAtoms:{A1:'N,CA,C,O'},allowRealignment:true,steps:21},
 {name:'partial',mode:'motif',structure,contig:'A1',fixedAtoms:false,partialT:5,steps:21},
 {name:'symmetry',length:12,symmetry:{id:'C2'},steps:21}];
 for(const precision of ['fp32','fp16','int8']){
  const baseURL=new URL('/weights/'+precision+'/',location.href),manifest=await(await fetch(new URL('manifest.json',baseURL))).json();
  const engine=await createRFD3Engine({manifest,architecture,baseURL,cache:'none'});
  try{
   for(const spec of cases){if(precision!=='fp16'&&!['monomer','motif'].includes(spec.name))continue;
    engine.trim();const input=buildInput(spec),features=input.features,handle=await engine.initialize(features),x=features.motif_pos.slice();
    await engine.run({features,X_noisy_L:x,t:16,n_recycle:2,initializer:handle});const warm=[];
    for(let i=0;i<3;i++){const t=performance.now();await engine.run({features,X_noisy_L:x,t:16,n_recycle:2,initializer:handle});warm.push(performance.now()-t);}
    engine.releaseInitializer(handle);
    const start=performance.now(),out=await engine.generate({...spec,n_recycle:2,seed:20261123,retainTrajectory:false}),wallSeconds=(performance.now()-start)/1000;
    let squared=0,count=0;for(let i=0;i<features.is_motif_atom_with_fixed_coord.length;i++)if(features.is_motif_atom_with_fixed_coord[i]){count++;for(let c=0;c<3;c++)squared+=(out.X_L.data[i*3+c]-features.motif_pos[i*3+c])**2;}
    results.push({precision,case:spec.name,tokens:features.residue_index.length,atoms:features.atom_to_token_map.length,updates:out.schedule.length-1,warmStepMs:warm,warmStepMeanMs:warm.reduce((a,b)=>a+b,0)/warm.length,wallSeconds,peakBytes:engine.peakMemoryUsage(),finite:out.X_L.data.every(Number.isFinite),motifRmsd:count?Math.sqrt(squared/count):null,seed:out.seed,motifAssignments:out.motifAssignments,pdb:out.pdb,mmcif:out.mmcif});
   }
  }finally{engine.destroy();}
 }
 return {results};
}
run().then(postMessage,error=>postMessage({error:error.stack}));
