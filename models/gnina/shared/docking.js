import {parseReceptor,parseLigands} from "./structure.js";
import {checkAbort} from "./loader.js";
import {vinaBreakdown} from "./scoring.js";
import {buildTorsionTree,coordinates,dockingPose} from "./torsions.js";
import {buildVinaGrids,dockingBox,pairwiseObjective,yieldTask} from "./vina-grid.js";
import {dockingObjective,localMinimizer,intramolecular,rmsd,searchChain} from "./docking-search.js";

export {buildTorsionTree,dockingPose,coordinates,buildVinaGrids,dockingBox};
export {dockingObjective,randomConformation,seededRandom} from "./docking-search.js";
export {sampleVinaGrid} from "./vina-grid.js";

function singleLigand(lig,format) {
  if(typeof lig!=="string") return lig;
  const poses=parseLigands(lig,format);
  if(poses.length!==1) throw Error("dockPose requires one ligand");
  return poses[0];
}
const integer=(n,min=1)=>Number.isSafeInteger(n)&&n>=min;

async function search(tree,grid,options,check,progress) {
  const {exhaustiveness}=options,banks=new Array(exhaustiveness);
  let completed=0;
  function save(id,bank) {
    banks[id]=bank;progress("search",++completed,exhaustiveness);
  }
  progress("search",0,exhaustiveness);
  if(typeof Worker==="undefined" || options.workers===1) {
    for(let id=0;id<exhaustiveness;id++) save(id,await searchChain(tree,grid,options,id,check));
  } else {
    const pool=[];
    let timer,next=0;
    try {
      await new Promise((resolve,reject)=>{
        // Also observe engine destruction and creation-signal cancellation
        // while all CPU work runs off the browser's main thread.
        timer=setInterval(()=>{try{check();}catch(e){reject(e);}},30);
        const mapBytes=[...grid.grids,...tree.tables].reduce((s,t)=>s+t.byteLength,0);
        const memoryLimit=Math.max(1,Math.floor(256*1024**2/mapBytes));
        for(let i=0;i<Math.min(options.workers,exhaustiveness,memoryLimit);i++) {
          const worker=new Worker(new URL("./docking-worker.js",import.meta.url),{type:"module"});
          pool.push(worker);
          worker.onerror=e=>reject(Error(e.message));
          worker.onmessage=({data})=>{
            try {
              if(data.error) throw Error(data.error);
              save(data.id,data.bank);
              if(next<exhaustiveness) worker.postMessage({id:next++});
              if(completed===exhaustiveness) resolve();
            } catch(e){reject(e);}
          };
          worker.postMessage({id:next++,context:{tree,grid,options}});
        }
      });
    } finally {clearInterval(timer);for(const worker of pool) worker.terminate();}
  }
  const bank=[];
  // Merge deterministically regardless of worker completion order.
  const minima=banks.flat().sort((a,b)=>a.energy-b.energy);
  // GNINA uses a 2 Å cross-chain merge before final filtering.
  for(const v of minima) {
    if(bank.some(b=>rmsd(b.coords,v.coords)<Math.max(2,options.minRmsd))) continue;
    bank.push(v);
    if(bank.length===options.limit) break;
  }
  for(const v of minima) {
    if(bank.length===options.limit) break;
    if(!bank.some(b=>b===v || rmsd(b.coords,v.coords)<options.minRmsd)) bank.push(v);
  }
  return bank;
}

function refinementPenalty(tree,p,force,box) {
  force.fill(0);
  let value=intramolecular(tree,p,force);
  for(let i=0;i<p.length;i++){
    const d=i%3,excess=Math.max(0,Math.abs(p[i]-box.center[d])-box.size[d]/2);
    value+=10*excess*excess;
  }
  return value;
}

// The CNN backend exposes forward scores. Pattern-search its loss in all
// pose coordinates, retaining intraligand repulsion and docking box restraints.
async function refine(tree,box,q,initial,score) {
  const coords=new Float64Array(3*tree.ligand.atoms.length),force=coords.slice();
  const loss=s=>-Math.log(Math.max(1e-30,s.cnnScore))+
    refinementPenalty(tree,coordinates(tree,q,coords,false),force,box);
  let best=initial,value=loss(initial);
  for(const scale of [1,.5]) for(let d=0;d<q.length;d++) {
    const start=q[d],delta=(d<3?.25:.1)*scale;
    let selected=start;
    for(const sign of [-1,1]){
      q[d]=start+sign*delta;
      const result=await score(dockingPose(tree,q)),v=loss(result);
      if(v<value){best=result;value=v;selected=q[d];}
    }
    q[d]=selected;
  }
  return best;
}

export async function dockPose(receptor,ligand,options={},scorePose,checkEngine=()=>{}) {
  const {signal,onProgress,seed=42,exhaustiveness=8,numModes=9,minRmsd=1,cnnScoring="rescore",
    rescoreTopK=Math.max(numModes*2,16),sortBy="cnnScore",workers=Math.min(8,globalThis.navigator?.hardwareConcurrency??1)}=options;
  const check=()=>{checkAbort(signal);checkEngine(signal);};
  const progress=(phase,completed,total)=>{check();onProgress?.({phase,completed,total});check();};
  check();
  if(![exhaustiveness,numModes,rescoreTopK,workers].every(n=>integer(n)) || !Number.isSafeInteger(seed) ||
      !Number.isFinite(minRmsd)||minRmsd<0 || !["cnnScore","cnnVS","cnnAffinity","vina"].includes(sortBy) ||
      !["rescore","refinement","none"].includes(cnnScoring)) throw Error("Invalid docking options");
  if(cnnScoring!=="none" && !scorePose) throw Error("CNN docking requires an initialized scoring engine");
  if(typeof receptor==="string") receptor=parseReceptor(receptor,options.receptorFormat);
  ligand=singleLigand(ligand,options.ligandFormat);
  const t0=performance.now(),tree=buildTorsionTree(ligand);
  const reference=singleLigand(options.autoboxLigand,options.ligandFormat);
  const box=dockingBox(receptor,reference,{...options,autoboxLigand:reference});
  const steps=options.steps??(80+400*tree.rotors.length);
  if(!integer(steps)) throw Error("Invalid Monte Carlo step count");
  const grid=await buildVinaGrids(receptor,ligand,box,{spacing:options.spacing,signal,check,onProgress});
  const t1=performance.now();
  const bank=await search(tree,grid,{seed,exhaustiveness,minRmsd,steps,workers,limit:rescoreTopK},check,progress);
  const t2=performance.now(),ranked=[];
  const score=pose=>{check();return scorePose(receptor,pose,{signal,contributions:true});};
  progress("rescore",0,bank.length);
  const polish=localMinimizer(dockingObjective(tree,grid,pairwiseObjective(receptor,tree,box)),6+tree.rotors.length);
  for(const {q} of bank) {
    if(cnnScoring==="none"){await yieldTask();check();}
    polish(q,50);
    let pose=dockingPose(tree,q),result;
    if(cnnScoring==="none"){
      const contributions=vinaBreakdown(receptor,pose);
      result={contributions,vinaScore:contributions.vinaScore,cnnScore:null,cnnAffinity:null,cnnVS:null,models:{}};
    } else {
      result=await score(pose);
      if(cnnScoring==="refinement"){result=await refine(tree,box,q,result,score);pose=dockingPose(tree,q);}
    }
    ranked.push({pose,...result,coords:coordinates(tree,q,new Float64Array(ligand.atoms.length*3),false)});
    progress("rescore",ranked.length,bank.length);
  }
  ranked.sort((a,b)=>sortBy==="vina"||cnnScoring==="none"?a.vinaScore-b.vinaScore:b[sortBy]-a[sortBy]);
  const poses=[],selected=[];
  for(const {coords,...v} of ranked) {
    if(selected.some(p=>rmsd(p,coords)<minRmsd)) continue;
    selected.push(coords);
    poses.push({...v,rank:poses.length+1,rmsdToBest:rmsd(coords,selected[0])});
    if(poses.length===numModes) break;
  }
  return {poses,box,stats:{gridMs:t1-t0,searchMs:t2-t1,rescoreMs:performance.now()-t2,
    evaluatedCandidates:exhaustiveness*steps,rotors:tree.rotors.length}};
}
