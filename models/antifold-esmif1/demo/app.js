import {createGVPTransformerEngine} from '../webgpu/index.js';
import {parseStructure,combineChains,regionMask,VOCAB} from '../shared/index.js';
const $=id=>document.getElementById(id);let engine,key,controller,result,lastInput,structureText='',structureFormat='pdb';
const status=text=>{$('status').textContent=text;};
const split=text=>text.split(/[\s,]+/).map(s=>s.trim()).filter(Boolean);
function download(name,text,type='text/plain'){const a=document.createElement('a'),url=URL.createObjectURL(new Blob([text],{type}));a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
async function example(file,variant,chains,heavy='',light=''){
 try{const response=await fetch(file);if(!response.ok)throw new Error(`Example unavailable (${response.status})`);$('structure').value=await response.text();$('variant').value=variant;$('chains').value=chains;$('heavy').value=heavy;$('light').value=light;$('target').value='';$('sequence').value='';$('regions').value=variant==='antifold'?'CDRH3'+(light?', CDRL3':''):'';status('Example loaded. Ready to run.');}catch(e){status(e.message);}
}
$('example-monomer').onclick=()=>example('../fixtures/1l2y.pdb','esmif1','A');
$('example-antibody').onclick=()=>example('../data/pdbs/6y1l_imgt.pdb','antifold','H, L','H','L');
$('example-vhh').onclick=()=>example('../data/nanobody/8oi2_imgt.pdb','antifold','B','B');
$('file').onchange=async()=>{const file=$('file').files[0];if(file){$('structure').value=await file.text();status(`${file.name} loaded.`);}};
$('cancel').onclick=()=>controller?.abort();
$('release').onclick=()=>{engine?.destroy();engine=null;key=null;status('GPU released.');};
$('run').onclick=async()=>{
 controller=new AbortController();$('run').disabled=true;$('cancel').disabled=false;$('release').disabled=true;const started=performance.now();
 try{
  structureText=$('structure').value;const parsed=parseStructure(structureText);structureFormat=parsed.format;
  const chains=split($('chains').value),opts={...(chains.length?{chains}:{}),heavyChain:$('heavy').value.trim()||undefined,lightChain:$('light').value.trim()||undefined,targetChain:$('target').value.trim()||undefined,numbering:$('numbering').value};
  const input=combineChains(parsed,opts);Object.assign(input,opts);lastInput=input;
  const regions=split($('regions').value);if(regions.length)opts.regions=regions;
  const variant=$('variant').value,precision=$('precision').value,nextKey=`${variant}/${precision}`;
  if(!engine||key!==nextKey){engine?.destroy();engine=null;key=null;status('Loading model weights…');
   engine=await createGVPTransformerEngine({baseUrl:new URL('../weights',import.meta.url).pathname,variant,precision,cache:'cache',signal:controller.signal,onProgress:p=>{if(p.phase==='weights'){$('progress').value=p.loaded/p.total;status(`Loading model · ${Math.round(100*p.loaded/p.total)}%`);}else if(p.phase==='inference'){$('progress').removeAttribute('value');status('Computing sequence preferences…');}}});key=nextKey;
  }
  const common={...opts,signal:controller.signal},mode=$('mode').value;let sample;
  if(mode==='score'){
   const sequence=$('sequence').value.replace(/\s/g,'');result=await engine.predict(input,{...common,...(sequence?{sequence}:{})});result.tokens=sequence?sequence.match(/<[^>]+>|./g).map(t=>VOCAB.indexOf(t)):Array.from(input.sequence,t=>VOCAB.indexOf(t));
  }else{
   let length=input.sequence.length;if(mode==='autoregressive'&&opts.targetChain)length=input.residues.findIndex(r=>!r||r.chain!==opts.targetChain);if(length<1)length=input.sequence.length;
   const fixed={};for(const item of split($('fixed').value)){const m=/^([^:]+):(-?\d+)([A-Za-z]?)=([A-Z])$/.exec(item);if(!m)throw new Error(`Invalid fixed residue: ${item}`);const chain=m[1]==='H'?opts.heavyChain:m[1]==='L'?opts.lightChain:m[1],index=input.residues.findIndex(r=>r&&r.chain===chain&&r.number===+m[2]&&r.insertion===m[3]);if(index<0||index>=length)throw new Error(`Fixed residue is outside the sampled chain: ${item}`);fixed[index]=m[4];}
   if(mode==='autoregressive'){
    const mask=regions.length?regionMask(input.residues,opts):input.residues.map(r=>r?1:0);for(let i=0;i<length;i++)if(!mask[i]&&fixed[i]===undefined)fixed[i]=input.sequence[i];
   }
   const u=crypto.getRandomValues(new Uint32Array(length)),uniforms=Float32Array.from(u,x=>(x>>>8)/16777216),temperature=Number($('temperature').value),order=Array.from({length},(_,i)=>i);
   // Fisher-Yates permutation is recorded explicitly, not inferred from a seed.
   for(let i=order.length-1;i>0;i--){const draw=crypto.getRandomValues(new Uint32Array(1))[0],j=Math.floor(draw/4294967296*(i+1));[order[i],order[j]]=[order[j],order[i]];}
   sample=await engine[mode==='independent'?'sampleIndependent':'sample'](input,{...common,uniforms,temperature,fixed,length,limitExpectedVariation:$('limit').checked,backmutationOrder:order});
   const full=Array.from(input.sequence,t=>VOCAB.indexOf(t));full.splice(0,sample.tokens.length,...sample.tokens);result=await engine.predict(input,{...common,...(mode==='autoregressive'?{sequence:full}:{})});result.tokens=full;result.sample=sample;
  }
  result.scoreKind=mode==='independent'?'native':'candidate';document.querySelector('.metrics small').textContent=mode==='independent'?'Native mean log likelihood':'Mean log likelihood';
  $('empty').hidden=true;$('result').hidden=false;$('result-sequence').textContent=result.tokens.map(t=>VOCAB[t]).join('');$('likelihood').textContent=result.logLikelihood.toFixed(4);$('perplexity').textContent=result.perplexity.toFixed(3);$('duration').textContent=`${((performance.now()-started)/1000).toFixed(2)} s`;
  const canvas=$('plot'),ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height;ctx.clearRect(0,0,W,H);ctx.fillStyle='#f2f7f3';ctx.fillRect(0,0,W,H);ctx.strokeStyle='#dce6df';for(let y=0;y<4;y++){ctx.beginPath();ctx.moveTo(0,y*H/4);ctx.lineTo(W,y*H/4);ctx.stroke();}
  const L=input.sequence.length;ctx.fillStyle='#32876a';for(let i=0;i<L;i++){const p=Math.max(...result.canonicalProbabilities.subarray(i*20,(i+1)*20));if(input.residues[i])ctx.fillRect(i*W/L,H-p*(H-12),Math.max(1,W/L-1),p*(H-12));}
  $('progress').value=1;status('Complete. Results are ready to download.');
 }catch(e){status(e.name==='AbortError'?'Cancelled. GPU scratch memory released.':`Unable to run: ${e.message}`);$('progress').value=0;}
 finally{$('run').disabled=false;$('cancel').disabled=true;$('release').disabled=false;}
};
$('fasta').onclick=()=>{if(!result)return;const chains=new Map();lastInput.residues.forEach((r,i)=>{if(r)chains.set(r.chain,(chains.get(r.chain)??'')+VOCAB[result.tokens[i]]);});download('designed-sequences.fasta',[...chains].map(([c,s])=>`>chain_${c}\n${s}`).join('\n')+'\n');};
$('csv').onclick=()=>{if(!result)return;const scored=typeof result.sequence==='string'?result.sequence.match(/<[^>]+>|./g):Array.from(result.sequence,t=>typeof t==='number'?VOCAB[t]:t);const lines=['chain,residue,insertion,native,designed_sequence,scored_residue,log_probability,entropy_perplexity'];lastInput.residues.forEach((r,i)=>{if(r)lines.push([r.chain,r.number,r.insertion,lastInput.sequence[i],VOCAB[result.tokens[i]],scored[i],result.logProbabilities[i],result.entropyPerplexity[i]].join(','));});download('residue-scores.csv',lines.join('\n')+'\n','text/csv');};
$('download-structure').onclick=()=>download(`input.${structureFormat}`,structureText);
