/** PDB/mmCIF model-1 peptide backbone parsing; residue identity is never an index. */
import {PEPTIDE_NAMES,RESIDUE_LETTERS as AA} from './residue-names.js';
const empty = v => v == null || v === '.' || v === '?' ? '' : v.trim();
function cifTokens(text) {
  const result=[];let i=0,lineStart=true;
  while(i<text.length){
    if(/\s/.test(text[i])){lineStart=text[i]==='\n'?true:lineStart;i++;continue;}
    if(text[i]==='#'){while(i<text.length&&text[i]!=='\n')i++;continue;}
    if(text[i]===';'&&lineStart){const a=++i;let end=text.indexOf('\n;',i);if(end<0)throw new Error('Unterminated CIF multiline value');result.push(text.slice(a,end));i=end+2;lineStart=false;continue;}
    lineStart=false;
    if(text[i]==="'"||text[i]==='"'){const q=text[i++],a=i;while(i<text.length&&!(text[i]===q&&(i+1===text.length||/\s/.test(text[i+1]))))i++;if(i===text.length)throw new Error('Unterminated CIF quote');result.push(text.slice(a,i++));}
    else {const a=i;while(i<text.length&&!/\s/.test(text[i]))i++;result.push(text.slice(a,i));}
  }
  return result;
}
function readCIF(text) {
  const t=cifTokens(text),atoms=[];
  for(let i=0;i<t.length;i++)if(t[i]==='loop_'){
    const cols=[];while(t[i+1]?.startsWith('_'))cols.push(t[++i]);
    const atom=cols[0]?.startsWith('_atom_site.');if(!cols.length)throw new Error('Empty CIF loop');
    const idx=Object.fromEntries(cols.map((c,j)=>[c.replace('_atom_site.',''),j]));
    while(i+1<t.length&&!/^(loop_|stop_|data_|save_|_)/.test(t[i+1])){
      const row=t.slice(i+1,i+1+cols.length);if(row.length!==cols.length)throw new Error('Truncated CIF loop');i+=cols.length;
      if(!atom)continue;
      const get=(name,fallback)=>row[idx[name]??idx[fallback]];
      atoms.push({group:get('group_PDB')??'ATOM',model:Number(get('pdbx_PDB_model_num')??1),atom:empty(get('auth_atom_id','label_atom_id')),name:empty(get('auth_comp_id','label_comp_id')),chain:empty(get('auth_asym_id','label_asym_id')),number:Number(get('auth_seq_id','label_seq_id')),insertion:empty(get('pdbx_PDB_ins_code')),alt:empty(get('label_alt_id')),xyz:['Cartn_x','Cartn_y','Cartn_z'].map(k=>Number(get(k)))});
    }
  }
  return atoms;
}
function readPDB(text) {
  let model=1,seen=false;const atoms=[];
  for(const line of text.split(/\r?\n/)){
    const rec=line.slice(0,6).trim();if(rec==='MODEL'){if(seen)break;seen=true;model=1;continue;}if(rec==='ENDMDL')break;
    if(rec!=='ATOM'&&rec!=='HETATM')continue;
    atoms.push({group:rec,model,atom:line.slice(12,16).trim(),alt:line.slice(16,17).trim(),name:line.slice(17,20).trim(),chain:line.slice(21,22).trim(),number:Number(line.slice(22,26)),insertion:line.slice(26,27).trim(),xyz:[30,38,46].map(i=>Number(line.slice(i,i+8)))});
  }
  return atoms;
}
export function parseStructure(text,{format,chains}={}) {
  if(typeof text!=='string'||!text.trim())throw new Error('Structure text is empty');
  format??=/^\s*data_/m.test(text)?'cif':'pdb';if(!['cif','mmcif','pdb'].includes(format))throw new Error('Expected PDB or mmCIF');
  const atoms=format==='pdb'?readPDB(text):readCIF(text),groups=new Map(),diagnostics={ignoredAtoms:0};
  const firstModel=atoms[0]?.model;
  for(const a of atoms){
    if(a.model!==firstModel)continue;
    const key=JSON.stringify([a.chain,a.number,a.insertion,a.name]);
    if(!groups.has(key))groups.set(key,{...a,atoms:[],alt:''});
    const g=groups.get(key);if(a.alt&&!g.alt)g.alt=a.alt;g.atoms.push(a);
  }
  const result=new Map();
  for(const g of groups.values()){
    if(chains&&!chains.includes(g.chain))continue;
    const bb=g.atoms.filter(a=>(!a.alt||a.alt===g.alt)&&['N','CA','C'].includes(a.atom));
    if(!bb.length||!PEPTIDE_NAMES.has(g.name)){diagnostics.ignoredAtoms+=g.atoms.length;continue;}
    if(!AA[g.name])throw new Error(`Unsupported residue ${g.name} at ${g.chain}:${g.number}${g.insertion}`);
    if(!Number.isInteger(g.number))throw new Error('Invalid residue number');
    const coords=new Float32Array(9).fill(NaN),seen=new Set();
    for(const a of bb){if(seen.has(a.atom))throw new Error(`Duplicate backbone atom ${a.atom}`);seen.add(a.atom);if(a.xyz.some(v=>!Number.isFinite(v)))throw new Error('Invalid coordinate');coords.set(a.xyz,['N','CA','C'].indexOf(a.atom)*3);}
    if(!result.has(g.chain))result.set(g.chain,{id:g.chain,residues:[],sequence:'',coords:[]});
    const c=result.get(g.chain);c.residues.push({chain:g.chain,number:g.number,insertion:g.insertion,name:g.name});c.sequence+=AA[g.name];c.coords.push(...coords);
  }
  if(!result.size)throw new Error('No peptide backbone found');
  for(const id of chains??[])if(!result.has(id))throw new Error(`Chain ${id} not found`);
  for(const c of result.values())c.coords=new Float32Array(c.coords);
  return {chains:[...result.values()],diagnostics,format};
}
export function combineChains(structure,{chains,targetChain,heavyChain,lightChain}={}) {
  let order=chains??structure.chains.map(c=>c.id);
  if(new Set(order).size!==order.length)throw new Error('Duplicate chain selection');
  if(heavyChain){if(!order.includes(heavyChain)||lightChain&&!order.includes(lightChain))throw new Error('Antibody chain is not selected');order=[heavyChain,...(lightChain?[lightChain]:[]),...order.filter(c=>c!==heavyChain&&c!==lightChain)];}
  if(targetChain){if(!order.includes(targetChain))throw new Error('Target chain is not selected');order=[targetChain,...order.filter(c=>c!==targetChain)];}
  const coords=[],residues=[];let sequence='';
  for(const [i,id] of order.entries()){
    const c=structure.chains.find(c=>c.id===id);if(!c)throw new Error(`Chain ${id} not found`);
    if(i){coords.push(...new Array(90).fill(NaN));residues.push(...new Array(10).fill(null));sequence+='----------';}
    coords.push(...c.coords);residues.push(...c.residues);sequence+=c.sequence;
  }
  return {coords:Float32Array.from(coords),residues,sequence,targetChain};
}
export const VOCAB=['<null_0>','<pad>','<eos>','<unk>',...'LAGVSERTIDPKQNFYMHWCXBUZO','.', '-', '<null_1>','<mask>','<cath>','<af2>'];
export const CANONICAL='ACDEFGHIKLMNPQRSTVWY';
export const CANONICAL_TOKENS=Int32Array.from(CANONICAL,c=>VOCAB.indexOf(c));
export function tokenize(sequence) {
  const parts=typeof sequence==='string'?sequence.replace(/\s/g,'').match(/<[^>]+>|./g):sequence;
  if(!parts?.length)throw new Error('Sequence is empty');
  return Int32Array.from(parts,t=>{const i=typeof t==='number'?t:VOCAB.indexOf(t);if(!Number.isInteger(i)||i<0||i>=VOCAB.length)throw new Error(`Invalid sequence token: ${t}`);return i;});
}
export function prepareInput(input,options={}) {
  if(typeof input==='string')input=combineChains(parseStructure(input,options),options);
  else if(!input?.coords&&input?.chains)input=combineChains(input,options);
  if(!input?.coords)throw new Error('Expected backbone coordinates or a PDB/mmCIF structure');
  const raw=input.coords instanceof Float32Array?input.coords:Float32Array.from(input.coords.flat(2),x=>x===null?NaN:x);
  const L=raw.length/9;if(!Number.isInteger(L)||L<1||L>2048)throw new Error('Expected 1–2048 backbone residues');
  const sequence=options.sequence??input.sequence??'X'.repeat(L),tokens=tokenize(sequence);
  if(tokens.length!==L)throw new Error('Sequence and structure lengths differ');
  const coords=new Float32Array((L+2)*9).fill(Infinity);coords.set(raw,9);
  const padding=new Int32Array(L+2),coordMask=new Int32Array(L+2),confidence=new Float32Array(L+2);
  const cf=options.confidence??input.confidence??1;
  if(typeof cf!=='number'&&cf.length!==L)throw new Error('Confidence length mismatch');
  for(let i=0;i<L+2;i++){
    padding[i]=Number.isNaN(coords[i*9])?1:0;coordMask[i]=coords.subarray(i*9,(i+1)*9).every(Number.isFinite)?1:0;
    const c=i===0||i===L+1?-1:typeof cf==='number'?cf:cf[i-1];if(!Number.isFinite(c)||(i>0&&i<=L&&(c<0||c>1)))throw new Error('Confidence must be between zero and one');
    confidence[i]=c*coordMask[i]-padding[i];
  }
  if(!coordMask.some(Boolean))throw new Error('At least one complete N/CA/C residue is required');
  if(input.residues&&input.residues.length!==L)throw new Error('Residue metadata length mismatch');
  const prev=new Int32Array(L);prev[0]=33;prev.set(tokens.subarray(0,L-1),1);
  return {L,N:L+2,coords,padding,coordMask,confidence,tokens,prev,sequence,residues:input.residues??Array.from({length:L},(_,i)=>({chain:'A',number:i+1,insertion:''})),...Object.fromEntries(['heavyChain','lightChain','targetChain','numbering','regions'].map(k=>[k,options[k]??input[k]]))};
}
export function regionMask(residues,{regions=['CDR1','CDR2','CDR3'],numbering='imgt',heavyChain,lightChain}={}) {
  if(!['imgt','kabat'].includes(numbering))throw new Error('Specify imgt or kabat numbering');
  if(!heavyChain)throw new Error('Region selection requires an explicit heavy chain');
  const bounds=numbering==='imgt'?{H:[[27,38],[56,65],[105,117]],L:[[27,38],[56,65],[105,117]]}:{H:[[31,35],[50,65],[95,102]],L:[[24,34],[50,56],[89,97]]};
  const selectors=regions.map(region=>{
    let m=/^([^:]+):(-?\d+[A-Za-z]?(?:--?\d+)?(?:,-?\d+[A-Za-z]?(?:--?\d+)?)*)$/.exec(region);
    if(m){const chain=m[1]==='H'?heavyChain:m[1]==='L'?lightChain:m[1];const ranges=m[2].split(',').map(p=>{const r=/^(-?\d+)([A-Za-z]?)(?:-(-?\d+))?$/.exec(p);const a=+r[1],b=r[3]===undefined?a:+r[3];if(b<a)throw new Error('Reversed residue range');return {a,b,ins:r[2]};});return r=>r.chain===chain&&ranges.some(x=>r.number>=x.a&&r.number<=x.b&&(!x.ins||r.insertion===x.ins));}
    m=/^(all|CDR|FW)([HL]?)([1-4]?)$/.exec(region);if(!m)throw new Error(`Invalid region ${region}`);
    const [,type,role,num]=m;if(type==='CDR'&&num==='4')throw new Error('No CDR4');
    return r=>{
      const roleHere=r.chain===heavyChain?'H':r.chain===lightChain?'L':null;if(!roleHere||(role&&role!==roleHere))return false;
      const max=numbering==='imgt'?128:roleHere==='H'?113:107;if(r.number<1||r.number>max)return false;
      if(type==='all')return true;const b=bounds[roleHere];
      const regionsHere=type==='CDR'?b:[[1,b[0][0]-1],[b[0][1]+1,b[1][0]-1],[b[1][1]+1,b[2][0]-1],[b[2][1]+1,max]];
      return regionsHere.some(([a,b],i)=>(!num||i===+num-1)&&r.number>=a&&r.number<=b);
    };
  });
  return Int32Array.from(residues,r=>r&&selectors.some(s=>s(r))?1:0);
}
