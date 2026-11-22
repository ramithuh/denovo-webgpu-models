export const VOCAB=['<cls>','<pad>','<eos>','<unk>',...'LAGVSERTIDPKQNFYMHWCXBUZO','.', '-', '|','<mask>'];
export const SASA_BOUNDS=[.8,4,9.6,16.4,24.5,32.9,42,51.5,61.2,70.9,81.6,93.3,107.2,125.4,151.4];
export function tokenize(sequence,{specialTokens=true}={}) {
 if(typeof sequence!=='string') throw new TypeError('sequence must be a string');
 const s=sequence.replace(/\s/g,'').toUpperCase().replace(/:/g,'|');
 if(!s.length||!/^[A-Z._|\-]+$/.test(s))throw new Error('Invalid sequence');
 const t=Array.from(s,c=>c==='_'?32:VOCAB.indexOf(c));
 if(t.some(x=>x<0))throw new Error('Invalid sequence residue');
 return Int32Array.from(specialTokens?[0,...t,2]:t);
}
export function decodeSequence(tokens){return Array.from(tokens,t=>t===32?'_':t===31?'|':t>=4&&t<31?VOCAB[t]:'').join('');}
function flat(a){return ArrayBuffer.isView(a)?Array.from(a):a.flat(Infinity);}
export function prepareInput(input) {
 if(typeof input==='string')input={sequence:input};
 if(input.sequence_tokens&&Array.from(input.sequence_tokens).some(t=>!Number.isInteger(t)))throw new Error('Invalid sequence token');
 let tokens=input.sequence_tokens?Int32Array.from(input.sequence_tokens):input.sequence?tokenize(input.sequence):null;
 let coords=input.structure_coords??input.coordinates;
 if(coords)coords=Float32Array.from(flat(coords),x=>x==null?NaN:x);
 if(!tokens){const n=input.length??input.structure_tokens?.length??(coords?coords.length/9:0);if(!Number.isInteger(n)||n<1)throw new Error('Supply sequence, sequence_tokens, or length');tokens=new Int32Array(n).fill(32);}
 const L=tokens.length;
 if(L<1||L>1024)throw new RangeError('Token length must be 1–1024');
 if(Array.from(tokens).some(t=>t<0||t>32))throw new RangeError('Invalid sequence token');
 const track=(name,n,def,max,float=false)=>{
  const a=input[name];const out=float?new Float32Array(n):new Int32Array(n);
  if(a==null)out.fill(def);else if(typeof a==='number'){if(!float&&!Number.isInteger(a))throw new Error(`Invalid ${name}`);out.fill(a);}else {const v=flat(a);if(!float&&v.some(x=>!Number.isInteger(x)))throw new Error(`Invalid ${name}`);if(v.length!==n)throw new Error(`${name}: expected ${n} values`);out.set(v);}
  if(Array.from(out).some(v=>!Number.isFinite(v)||(!float&&(!Number.isInteger(v)||v<0||v>max))))throw new Error(`Invalid ${name}`);
  return out;
 };
 if(coords&&coords.length===(L-2)*9&&tokens[0]===0&&tokens[L-1]===2){const c=new Float32Array(L*9).fill(NaN);c.set(coords,9);coords=c;}
 if(coords&&coords.length!==L*9)throw new Error('Coordinates must contain N/CA/C xyz for every token or every residue');
 let ss=input.ss8_tokens;
 if(input.ss8){ss=[0,...Array.from(input.ss8,c=>{const t='GHITEBSC'.indexOf(c);if(t<0)throw new Error('Invalid ss8');return t+3;}),0];}
 let sa=input.sasa_tokens;
 if(input.sasa)sa=[0,...Array.from(input.sasa,v=>v==null?0:3+SASA_BOUNDS.filter(b=>v>b).length),0];
 input={...input,ss8_tokens:ss,sasa_tokens:sa};
 if(input.structure_tokens)input.structure_tokens=flat(input.structure_tokens).map(x=>x===-1?4096:x);
 const structure=track('structure_tokens',L,4096,4100);
 for(let i=0;i<L;i++)structure[i]=tokens[i]===0?4098:tokens[i]===2?4097:tokens[i]===1?4099:tokens[i]===31?4100:structure[i];
 return {L,sequence_tokens:tokens,structure_tokens:structure,ss8_tokens:track('ss8_tokens',L,0,10),sasa_tokens:track('sasa_tokens',L,0,18),
  function_tokens:track('function_tokens',L*8,0,259),residue_annotation_tokens:track('residue_annotation_tokens',L*16,0,1477),
  average_plddt:track('average_plddt',L,1,1,true),per_res_plddt:track('per_res_plddt',L,0,1,true),
  sequence_id:track('sequence_id',L,0,2147483647),chain_id:track('chain_id',L,0,2147483647),
  attention_mask:track('attention_mask',L,1,1),residue_index:input.residue_index?track('residue_index',L,0,2147483647):Int32Array.from({length:L},(_,i)=>i),
  structure_coords:coords??new Float32Array(L*9).fill(NaN),encodeStructure:!!coords&&!input.structure_tokens};
}
export function parsePDB(text,{chain}={}) {
 const residues=new Map();let activeChain=chain;
 for(const line of text.split(/\r?\n/)){
  if(line.startsWith('ENDMDL'))break;
  if(!line.startsWith('ATOM  '))continue;
  const c=line[21];if(activeChain===undefined)activeChain=c;if(c!==activeChain)continue;
  const atom=line.slice(12,16).trim(),alt=line[16];if(!['N','CA','C'].includes(atom)||![' ','A'].includes(alt))continue;
  const key=line.slice(22,27);if(!residues.has(key))residues.set(key,{name:line.slice(17,20).trim(),coords:new Float32Array(9).fill(NaN)});
  residues.get(key).coords.set([30,38,46].map(k=>Number(line.slice(k,k+8))),['N','CA','C'].indexOf(atom)*3);
 }
 if(!residues.size)throw new Error('No backbone atoms in requested chain');
 const aa={ALA:'A',ARG:'R',ASN:'N',ASP:'D',CYS:'C',GLN:'Q',GLU:'E',GLY:'G',HIS:'H',ILE:'I',LEU:'L',LYS:'K',MET:'M',PHE:'F',PRO:'P',SER:'S',THR:'T',TRP:'W',TYR:'Y',VAL:'V'};
 return {sequence:[...residues.values()].map(r=>aa[r.name]??'X').join(''),coordinates:Float32Array.from([...residues.values()].flatMap(r=>Array.from(r.coords)))};
}
