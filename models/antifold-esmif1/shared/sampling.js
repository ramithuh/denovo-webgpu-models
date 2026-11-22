import {prepareInput,regionMask,VOCAB,CANONICAL_TOKENS,tokenize} from './structure.js';
/** Native-conditioned AntiFold sampling, including its optional backmutation rule. */
export function independentPlan(input,options={}){
 const p=prepareInput(input,options),temperature=options.temperature??.2,uniforms=options.uniforms;
 if(!(temperature>0&&Number.isFinite(temperature)))throw new Error('Temperature must be positive');
 if(uniforms?.length!==p.L||Array.from(uniforms).some(u=>!(u>=0&&u<1)))throw new Error('Supply one uniform draw in [0,1) per input position');
 const selected=options.regions||p.regions?regionMask(p.residues,{...p,...options}):Int32Array.from(p.residues,r=>r?1:0);
 const fixed={};for(const [i,t]of Object.entries(options.fixed??{})){if(!Number.isInteger(+i)||+i<0||+i>=p.L)throw new Error('Invalid fixed position');const tok=tokenize([t])[0];fixed[i]=tok;selected[i]=0;}
 const order=options.backmutationOrder??Array.from({length:p.L},(_,i)=>i);
 if(options.limitExpectedVariation&&(order.length!==p.L||new Set(order).size!==p.L||order.some(i=>!Number.isInteger(i)||i<0||i>=p.L)))throw new Error('Backmutation order must be a permutation of residue indices');
 return {p,temperature,uniforms,selected,fixed,order};
}
export function finishIndependent(plan,draws,options={}){
 const {p,selected,fixed,order}=plan,tokens=p.tokens.slice(),mutated=[];let deviations=0;
 for(let i=0;i<p.L;i++){
  if(selected[i]){tokens[i]=CANONICAL_TOKENS[draws[2*i]];if(draws[2*i]!==draws[2*i+1])deviations++;}
  if(fixed[i]!==undefined)tokens[i]=fixed[i];
  if(selected[i]&&tokens[i]!==p.tokens[i])mutated.push(i);
 }
 if(options.limitExpectedVariation){let n=mutated.length-deviations;const set=new Set(mutated);for(const i of order)if(n>0&&set.has(i)){tokens[i]=p.tokens[i];n--;}}
 return {tokens,sequence:Array.from(tokens,t=>VOCAB[t]).join(''),selected};
}
