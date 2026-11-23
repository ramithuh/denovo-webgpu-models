import {checkAbort} from "./loader.js";
import {forward} from "./network.js";
import {prepareGrid,vinaBreakdown,transformPose,centroid} from "./scoring.js";
import {parseReceptor,parseLigands} from "./structure.js";

export function createScorer(backend,models,creationSignal) {
  let closed=false;
  function check(signal) {
    if(closed) throw Error("GNINA engine has been destroyed");
    checkAbort(creationSignal);checkAbort(signal);
  }
  async function scorePose(receptor,ligand,options={}) {
    const {signal,trace,center,contributions=false}=options;
    check(signal);
    if(typeof receptor==="string") receptor=parseReceptor(receptor,options.receptorFormat);
    if(typeof ligand==="string") {
      const poses=parseLigands(ligand,options.ligandFormat);
      if(poses.length!==1) throw Error("Use scorePoses for multiple poses");
      ligand=poses[0];
    }
    const input=prepareGrid(receptor,ligand,center), ops=backend.begin(()=>check(signal));
    try {
      const grid=ops.grid(input);
      if(trace) await trace("grid",await ops.read(grid));
      const perModel={};
      let score=0,affinity=0;
      for(const name of models) {
        check(signal);
        const result=await forward(ops,grid,name,trace);
        perModel[name]=result;score+=result.cnnScore;affinity+=result.cnnAffinity;
      }
      check(signal);
      const cnnScore=score/models.length, cnnAffinity=affinity/models.length;
      const breakdown=vinaBreakdown(receptor,ligand);
      return {cnnScore,cnnAffinity,cnnVS:cnnScore*cnnAffinity,vinaScore:breakdown.vinaScore,
        models:perModel,...(contributions?{contributions:breakdown}:{})};
    } finally {ops.finish();}
  }
  // Measure a true CNN ablation delta at a fixed grid center. Hiding the only
  // heavy atom leaves an empty channel group, as in the upstream density grid.
  async function maskContributions(rec,lig,{atomIndices=[],residues=[],...options}={}) {
    if(typeof rec==="string") rec=parseReceptor(rec,options.receptorFormat);
    if(typeof lig==="string") [lig]=parseLigands(lig,options.ligandFormat);
    const fixed={...options,center:options.center??centroid(lig)};
    const baseline=await scorePose(rec,lig,fixed);
    const deltas=result=>Object.fromEntries(["cnnScore","cnnAffinity","cnnVS"].map(k=>[k,baseline[k]-result[k]]));
    const atoms=[],perResidue=[];
    for(const index of new Set(atomIndices)){
      if(!lig.atoms.some(a=>a.index===index)) throw Error("Unknown ligand atom index "+index);
      const remaining=lig.atoms.filter(a=>a.index!==index);
      atoms.push({index,...deltas(await scorePose(rec,{...lig,atoms:remaining},fixed))});
    }
    for(const residue of new Set(residues)){
      if(!rec.atoms.some(a=>a.residue===residue)) throw Error("Unknown receptor residue "+residue);
      const remaining=rec.atoms.filter(a=>a.residue!==residue);
      perResidue.push({residue,...deltas(await scorePose({...rec,atoms:remaining},lig,fixed))});
    }
    return {baseline,atoms,residues:perResidue};
  }
  async function scorePoses(rec,poses,options={}) {
    if(typeof rec==="string") rec=parseReceptor(rec,options.receptorFormat);
    if(typeof poses==="string") poses=parseLigands(poses,options.ligandFormat);
    const results=[];
    for(const pose of poses) results.push(await scorePose(rec,pose,options));
    return results;
  }
  async function minimizePose(rec,lig,options={}) {
    const {iterations=4,translationStep=.5,rotationStep=.12,cnnWeight=1}=options;
    if(!Number.isInteger(iterations)||iterations<0||![translationStep,rotationStep,cnnWeight].every(x=>Number.isFinite(x)&&x>=0)) throw Error("Invalid minimization options");
    if(typeof rec==="string") rec=parseReceptor(rec,options.receptorFormat);
    if(typeof lig==="string") [lig]=parseLigands(lig,options.ligandFormat);
    const energy=s=>s.vinaScore-cnnWeight*Math.log(Math.max(s.cnnScore,1e-30));
    let pose=lig,scores=await scorePose(rec,lig,options),step=1;
    for(let i=0;i<iterations;i++) {
      let best=pose;
      for(let axis=0;axis<6;axis++) for(const sign of [-1,1]) {
        const rotation=[0,0,0],shift=[0,0,0];
        (axis<3?shift:rotation)[axis%3]=sign*step*(axis<3?translationStep:rotationStep);
        const candidate=transformPose(pose,rotation,shift),s=await scorePose(rec,candidate,options);
        if(energy(s)<energy(scores)) {best=candidate;scores=s;}
      }
      if(best===pose) step*=.5;
      pose=best;
    }
    return {pose,scores,objective:energy(scores)};
  }
  return {scorePose,scorePoses,minimizePose,maskContributions,ensemble:[...models],adapterInfo:backend.adapterInfo,
    isFallbackAdapter:backend.isFallbackAdapter,memoryUsage:()=>backend.memoryUsage(),
    destroy(){if(!closed){closed=true;backend.destroy();}}};
}
