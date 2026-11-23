import {createGninaEngine,parseLigands,parseReceptor,buildTorsionTree,dockingBox,
  randomConformation,seededRandom,dockingPose} from "../webgpu/index.js";

// Enumerate chemical graph automorphisms for the fixture RMSD oracle. Aromatic
// bonds are equivalent across Kekulé representations. No spatial alignment.
function permutations(lig) {
  const n=lig.atoms.length,edges=Array.from({length:n},()=>new Uint8Array(n));
  const indices=new Map(lig.atoms.map((a,i)=>[a.index,i]));
  const aromatic=i=>[16,17].includes(lig.atoms[i].channel);
  for(const [u,v,order] of lig.bonds) {
    const i=indices.get(u),j=indices.get(v);
    if(i!==undefined&&j!==undefined) edges[i][j]=edges[j][i]=aromatic(i)&&aromatic(j)?4:order;
  }
  const labels=lig.atoms.map((a,i)=>[a.el,a.channel,a.donor,a.acceptor,a.charge??0,...edges[i].slice().sort()].join("/"));
  const result=[],p=[],used=new Set();
  function visit(i){
    if(i===n){result.push(p.slice());return;}
    for(let j=0;j<n;j++) if(labels[i]===labels[j]&&!used.has(j)&&p.every((v,k)=>edges[i][k]===edges[j][v])){
      p.push(j);used.add(j);visit(i+1);used.delete(j);p.pop();
    }
  }
  visit(0);return result;
}
function distance(a,b,p) {
  let s=0;
  for(let i=0;i<p.length;i++) for(const k of ["x","y","z"]) s+=(a.atoms[i][k]-b.atoms[p[i]][k])**2;
  return Math.sqrt(s/p.length);
}
export async function redock() {
  const engine=await createGninaEngine({baseUrl:"/weights/default",precision:"fp32",allowSoftwareFallback:false});
  const runs={};
  try {
    for(const [id,options] of [["184l",{}],["10gs",{}],["10gs",{rescoreTopK:48}]]) {
      const text=async type=>(await fetch("/fixtures/"+id+type)).text();
      const rec=parseReceptor(await text("_rec.pdb")),[native]=parseLigands(await text("_lig.sdf"));
      const box=dockingBox(rec,native),tree=buildTorsionTree(native);
      const scrambled=dockingPose(tree,randomConformation(tree,box,seededRandom(1907)));
      const maps=permutations(native);
      const distances=pose=>maps.map(p=>distance(pose,native,p));
      const start=performance.now();
      // Only six pocket bounds and a fully randomized conformer cross this boundary.
      const result=await engine.dockPose(rec,scrambled,{box,...options});
      const wallMs=performance.now()-start;
      const label=options.rescoreTopK?id+"_wide":id;
      const bestN=options.rescoreTopK?3:1;
      runs[label]={options,wallMs,stats:result.stats,box,inputRmsd:Math.min(...distances(scrambled)),
        poses:result.poses.map(({rank,pose,cnnScore,vinaScore,models})=>{
          if(Object.keys(models).length!==3) throw Error("Missing ensemble member");
          const ds=distances(pose);
          return {rank,cnnScore,vinaScore,rmsd:Math.min(...ds),orderedRmsd:ds[0]};
        })};
      const run=runs[label],best=Math.min(...run.poses.slice(0,bestN).map(p=>p.rmsd));
      if(run.inputRmsd<3 || !(best<2) || !(run.poses[0].cnnScore>.70) ||
        !run.poses.slice(0,bestN).some(p=>p.orderedRmsd<2))
        throw Error(label+" redocking failed: "+JSON.stringify(run));
      if(result.poses[0].rmsdToBest!==0 ||
        result.poses.some((p,i)=>p.rank!==i+1 || p.contributions.vinaScore!==p.vinaScore ||
          (i && p.cnnScore>result.poses[i-1].cnnScore))) throw Error("Invalid docked scores or ranks");
      if(id==="184l") {
        const short={box,steps:6,exhaustiveness:2,rescoreTopK:3,cnnScoring:"none"};
        const serial=await engine.dockPose(rec,scrambled,{...short,workers:1});
        const parallel=await engine.dockPose(rec,scrambled,{...short,workers:2});
        if(JSON.stringify(serial.poses)!==JSON.stringify(parallel.poses)) throw Error("Parallel seeded search diverged");
        const controller=new AbortController();
        let aborted=false;
        try {
          await engine.dockPose(rec,scrambled,{...short,steps:10000,workers:2,signal:controller.signal,
            onProgress:p=>{if(p.phase==="search"&&p.completed===0) setTimeout(()=>controller.abort(),10);}});
        } catch(error){if(error.name!=="AbortError") throw error;aborted=true;}
        if(!aborted || engine.memoryUsage().transientBytes!==0) throw Error("In-flight docking cancellation failed");
      }
    }
  } finally {engine.destroy();}
  const afterDestroy=engine.memoryUsage();
  if(afterDestroy.totalBytes!==0) throw Error("Docking leaked GPU buffers");
  return {adapter:engine.adapterInfo,precision:"fp32",ensemble:engine.ensemble,
    scrambleSeed:1907,searchSeed:42,afterDestroy,runs};
}
