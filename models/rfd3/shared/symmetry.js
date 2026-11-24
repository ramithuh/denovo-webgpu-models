// Pinned RFD3 atomwise symmetry convention: row vectors, entity-local ASU order.
export function validateSymmetry(f,{cfgScale=1,gamma0=0.6,symStepFrac=0.9}={}){
 if(cfgScale!==1)throw new Error('Symmetry sampling does not support CFG (cfgScale must equal 1)');
 if(!(gamma0>0.5)||!(symStepFrac>0&&symStepFrac<=1))throw new Error('Symmetry requires gamma0 > 0.5 and 0 < symStepFrac <= 1');
 const n=f.atom_to_token_map.length;
 for(const key of ['sym_entity_id','sym_transform_id','is_sym_asu'])if(f[key]?.length!==n)throw new Error('Missing symmetry feature '+key);
 if(!f.sym_transform)throw new Error('Missing symmetry transforms');
 for(const [id,value]of Object.entries(f.sym_transform)){
  if(+id===-1)continue;const [r,t]=value;if(r?.length!==9||t?.length!==3||![...r,...t].every(Number.isFinite))throw new Error('Invalid symmetry transform');
  for(let a=0;a<3;a++)for(let b=0;b<3;b++){let dot=0;for(let k=0;k<3;k++)dot+=r[a*3+k]*r[b*3+k];if(Math.abs(dot-Number(a===b))>1e-5)throw new Error('Symmetry rotation is not orthogonal');}
 }
 for(const entity of new Set(f.sym_entity_id))if(entity!==-1){const asu=Array.from(f.is_sym_asu,(v,i)=>v&&f.sym_entity_id[i]===entity?1:0).reduce((a,b)=>a+b,0);if(!asu)throw new Error('Symmetry entity has no ASU');for(const id of new Set(Array.from(f.sym_transform_id).filter((_,i)=>f.sym_entity_id[i]===entity))){if(!f.sym_transform[id])throw new Error('Missing symmetry frame');const size=Array.from(f.sym_entity_id,(e,i)=>Number(e===entity&&f.sym_transform_id[i]===id)).reduce((a,b)=>a+b,0);if(size!==asu)throw new Error('Symmetry subunits have unequal atom counts');}}
}
export function applySymmetry(coordinates,f,{partialDiffusion=false}={}){
 const n=coordinates.length/3,x=coordinates.slice(),center=[0,0,0],active=[];
 for(let i=0;i<n;i++)if(f.sym_entity_id[i]!==-1)active.push(i);
 if(!partialDiffusion&&active.length){for(const i of active)for(let c=0;c<3;c++)center[c]+=x[i*3+c]/active.length;for(const i of active)for(let c=0;c<3;c++)x[i*3+c]-=center[c];}
 const out=x.slice();
 for(const entity of new Set(f.sym_entity_id))if(entity!==-1){const asu=active.filter(i=>f.sym_entity_id[i]===entity&&f.is_sym_asu[i]);
  for(const id of new Set(active.filter(i=>f.sym_entity_id[i]===entity).map(i=>f.sym_transform_id[i]))){const target=active.filter(i=>f.sym_entity_id[i]===entity&&f.sym_transform_id[i]===id),[r,t]=f.sym_transform[id];for(let a=0;a<target.length;a++)for(let c=0;c<3;c++){let sum=t[c];for(let k=0;k<3;k++)sum+=x[asu[a]*3+k]*r[k*3+c];out[target[a]*3+c]=sum;}}
 }
 return out;
}
export function cyclicSymmetryFeatures(features,order){
 if(!Number.isInteger(order)||order<2)throw new Error('Symmetry order must be at least 2');
 const l=features.atom_to_token_map.length;if(l%order)throw new Error('Symmetric atom count is not divisible by order');const per=l/order;
 const sym_transform={};for(let i=0;i<order;i++){const a=2*Math.PI*i/order,c=Math.cos(a),s=Math.sin(a);sym_transform[i]=[[c,-s,0,s,c,0,0,0,1],[0,0,0]];}
 return {...features,sym_entity_id:new Float32Array(l),sym_transform_id:Float32Array.from({length:l},(_,i)=>Math.floor(i/per)),is_sym_asu:Float32Array.from({length:l},(_,i)=>Number(i<per)),sym_transform};
}
