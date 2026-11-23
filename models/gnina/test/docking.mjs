import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {parseReceptor,parseLigands,parseSDF,typeMolecule,buildTorsionTree,coordinates,dockingPose,
  buildVinaGrids,dockingObjective,sampleVinaGrid,dockingBox,randomConformation,seededRandom,vinaScore,dockPose} from "../shared/index.js";
import {createScorer} from "../shared/api.js";

const read=p=>readFile("fixtures/"+p,"utf8");
const sdf=await read("10gs_lig.sdf"),rec=parseReceptor(await read("10gs_rec.pdb")),[lig]=parseLigands(sdf);
const box=dockingBox(rec,lig),tree=buildTorsionTree(lig);
assert.equal(tree.rotors.length,13);
function bondLength(a,b){return Math.hypot(a.x-b.x,a.y-b.y,a.z-b.z);}
function checkGeometry(lig,moved) {
  const byIndex=atoms=>new Map(atoms.map(a=>[a.index,a]));
  const before=byIndex(lig.atoms),after=byIndex(moved.atoms);
  for(const [a,b] of lig.bonds) if(before.has(a)&&before.has(b))
    assert(Math.abs(bondLength(before.get(a),before.get(b))-bondLength(after.get(a),after.get(b)))<1e-5);
  lig.centerAtoms.forEach((a,i)=>{
    const parent=lig.centerParents[i],b=moved.centerAtoms[i];
    assert(Math.abs(bondLength(a,before.get(parent))-bondLength(b,after.get(parent)))<1e-5);
  });
}
const rng=seededRandom(1234);
for(let i=0;i<10;i++) checkGeometry(lig,dockingPose(tree,randomConformation(tree,box,rng)));
// Explicit polar H interspersed with heavy atoms exercises input-index remapping.
const raw=parseSDF(sdf)[0];
raw.atoms.unshift(lig.centerAtoms.at(-1));
const explicit=typeMolecule({atoms:raw.atoms,bonds:[[0,33,1],...raw.bonds.map(([i,j,o])=>[i+1,j+1,o])]},"ligand");
const et=buildTorsionTree(explicit);
checkGeometry(explicit,dockingPose(et,randomConformation(et,box,rng)));
const q=randomConformation(tree,box,rng);
const rebuilt=buildTorsionTree(dockingPose(tree,randomConformation(tree,box,rng)));
const a=coordinates(tree,q),b=coordinates(rebuilt,q);
assert(a.every((v,i)=>Math.abs(v-b[i])<1e-8),"Absolute pose depends on input orientation or dihedrals");

const grid=await buildVinaGrids(rec,lig,box);
const evaluate=dockingObjective(tree,grid),g=new Float64Array(q.length),scratch=g.slice();
let worst=0;
for(const angle of [0,.9,3.15]){
  q[3]=angle;q[4]=angle*.3;q[5]=-angle*.2;
  evaluate(q,g);
  for(let i=0;i<q.length;i++){
    const t=q[i],h=1e-6;
    q[i]=t+h;const hi=evaluate(q,scratch);
    q[i]=t-h;const lo=evaluate(q,scratch);
    q[i]=t;
    const fd=(hi-lo)/(2*h),error=Math.abs(fd-g[i])/Math.max(1e-6,Math.abs(fd),Math.abs(g[i]));
    worst=Math.max(worst,error);
    assert(error<1e-3,`DOF ${i}, analytic ${g[i]}, numerical ${fd}`);
  }
}
function gridAffinity(pose,points) {
  return sampleVinaGrid(grid,points,new Float64Array(points.length))/(1+.05846*pose.rotors);
}
const xyz=lig.atoms.flatMap(a=>[a.x,a.y,a.z]);
const snapped=xyz.map((v,i)=>{
  const d=i%3;return grid.origin[d]+Math.round((v-grid.origin[d])*grid.invStep[d])/grid.invStep[d];
});
const snapLig={...lig,atoms:lig.atoms.map((a,i)=>({...a,x:snapped[3*i],y:snapped[3*i+1],z:snapped[3*i+2]}))};
const nodeError=Math.abs(gridAffinity(lig,snapped)-vinaScore(rec,snapLig));
const nativeError=Math.abs(gridAffinity(lig,xyz)-vinaScore(rec,lig));
assert(nodeError<1e-4,"Affinity at lattice nodes: "+nodeError);
assert(nativeError<.6,"Off-lattice interpolation: "+nativeError);

const input={box,seed:42,steps:10,exhaustiveness:2,numModes:3,cnnScoring:"none"};
await assert.rejects(dockPose(rec,lig,{cnnScoring:"none"}),/box/);
const run=()=>dockPose(rec,lig,input);
const first=await run(),second=await run();
assert.deepEqual(first.poses,second.poses);
assert.equal(first.stats.evaluatedCandidates,20);
assert.equal(first.poses[0].rmsdToBest,0);
assert(first.poses.every((p,i)=>p.rank===i+1 && p.cnnScore===null && p.vinaScore===vinaScore(rec,p.pose)));
for(const phase of ["grid","search","rescore"]) {
  const ac=new AbortController();
  await assert.rejects(dockPose(rec,lig,{...input,signal:ac.signal,onProgress:p=>{if(p.phase===phase) ac.abort();}}),{name:"AbortError"});
}
const ac=new AbortController();
setTimeout(()=>ac.abort(),5);
await assert.rejects(dockPose(rec,lig,{...input,signal:ac.signal}),{name:"AbortError"});
assert.deepEqual(dockingBox(rec,lig,{pocketResidues:[rec.atoms[0].residue]}),
  dockingBox(rec,{atoms:rec.atoms.filter(a=>a.residue===rec.atoms[0].residue)}));
// Isolate the wall from intermolecular and internal terms at a box face.
const one={atoms:[lig.atoms[0]],rotors:0};
const empty=await buildVinaGrids({atoms:[]},one,{center:[0,0,0],size:[4,4,4]});
const wallGrad=new Float64Array(3);
for(const x of [1.999,2,2.001,3,-3]){
  const excess=Math.sign(x)*Math.max(0,Math.abs(x)-2);
  assert(Math.abs(sampleVinaGrid(empty,[x,0,0],wallGrad)-10*excess**2)<1e-12);
  assert(Math.abs(wallGrad[0]-20*excess)<1e-12);
}
// Exercise flexible CNN refinement and the score/pose correspondence with a
// cheap differentiable stand-in for the ensemble forward pass.
const toyRec={atoms:[{...lig.atoms[0],x:0,y:0,z:0}]};
const mockScore=async (r,pose)=>{
  const a=pose.atoms[0],cnnScore=1/(1+Math.abs(a.x)),cnnAffinity=a.y;
  return {cnnScore,cnnAffinity,cnnVS:cnnScore*cnnAffinity,vinaScore:vinaScore(r,pose),models:{}};
};
const toyLig=parseLigands(await read("184l_lig.sdf"))[0];
for(const sortBy of ["cnnScore","cnnVS","cnnAffinity","vina"]){
  const options={...input,center:[0,0,0],size:[10,10,10],box:undefined,
    steps:2,cnnScoring:"refinement",minRmsd:.1,sortBy};
  const {poses}=await dockPose(toyRec,toyLig,options,mockScore);
  for(const [i,p] of poses.entries()){
    checkGeometry(toyLig,p.pose);
    const key=sortBy==="vina"?"vinaScore":sortBy;
    assert.equal(p[key],(await mockScore(toyRec,p.pose))[key]);
    if(i) assert(sortBy==="vina"?poses[i-1][key]<=p[key]:poses[i-1][key]>=p[key]);
  }
}
const engine=createScorer({destroy(){}},[]);
engine.destroy();
await assert.rejects(engine.dockPose(rec,lig,input),/destroyed/);
console.log("Docking geometry, gradients, grid accuracy, reproducibility and cancellation passed.",
  {worstRelativeGradientError:worst,nodeError,nativeError});
