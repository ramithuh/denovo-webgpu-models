import { sha256 } from '../shared/loader.js';
export function assert(value,message) {if(!value)throw Error(message);}
export function metrics(actual,expected) {
  assert(actual.length===expected.length,'Tensor length mismatch');
  let ss=0,rr=0,maxAbs=0;
  for(let i=0;i<actual.length;i++){
    assert(Number.isFinite(actual[i])&&Number.isFinite(expected[i]),`Nonfinite value at ${i}`);
    const d=actual[i]-expected[i];ss+=d*d;rr+=expected[i]*expected[i];maxAbs=Math.max(maxAbs,Math.abs(d));
  }
  return {relRMS:Math.sqrt(ss/Math.max(rr,1e-60)),maxAbs};
}
export function compare(actual,expected,label,tolerance=1e-4) {
  const m=metrics(actual,expected);assert(m.relRMS<tolerance,`${label}: relRMS=${m.relRMS}, maxAbs=${m.maxAbs}`);return m;
}
export async function fixture(c,read) {
  const bytes=await read(c.file);assert(bytes.byteLength===c.byteLength&&await sha256(bytes)===c.sha256,'Fixture integrity failure');
  const out={};for(const [name,t] of Object.entries(c.tensors))out[name]=new Float32Array(bytes,t.offset,t.length);
  return out;
}
export function sampleOptions(c,ref) {
  const draws=Object.keys(ref).filter(k=>k.startsWith('draw.')).sort((a,b)=>+a.slice(5)-+b.slice(5)).map(k=>ref[k]);
  return {length:c.length,steps:c.steps,mode:c.mode,initialCa:ref.initial_ca,initialZ:ref.initial_z,draws,returnTrajectory:true};
}
export function checkFinalOutput(out,ref,c) {
  const result={};
  for(const [key,target] of [['ca','final_ca'],['z','final_z'],['coords','coords'],['logits','logits'],['maskedCoords','masked_coords']])result[key]=compare(out[key],ref[target],`${c.name}:${key}`);
  assert(out.residueTypes.every((x,i)=>x===ref.residue_types[i]),'Sequence argmax mismatch');
  assert(out.atomMask.every((x,i)=>x===ref.atom_mask[i]),'Atom mask mismatch');
  assert(out.pdb.includes('ATOM')&&out.pdb.endsWith('END\n'),'PDB output missing');
  return result;
}
export function checkOutput(out,ref,c) {
  const result=checkFinalOutput(out,ref,c);
  for(let i=0;i<c.steps;i++){
    const tr=out.trajectory[i];
    for(const [key,target] of [['ca',`state.${i}.bb_ca`],['z',`state.${i}.local_latents`],['velocityCa',`velocity.${i}.bb_ca`],['velocityZ',`velocity.${i}.local_latents`]])result[`step${i}.${key}`]=compare(tr[key],ref[target],`${c.name}:step${i}.${key}`);
  }
  return result;
}
