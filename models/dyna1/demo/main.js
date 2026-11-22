import {createDyna1Engine} from '../webgpu/index.js';
const $=id=>document.getElementById(id);let engine,controller,result,key;
$('cancel').onclick=()=>controller?.abort();
$('release').onclick=()=>{engine?.destroy();engine=null;key=null;$('release').disabled=true;$('status').textContent='Model released.';};
$('run').onclick=async()=>{
 controller=new AbortController();$('run').disabled=true;$('cancel').disabled=false;$('release').disabled=true;$('download').disabled=true;
 try{
  const baseUrl=new URL($('base').value.endsWith('/')?$('base').value:$('base').value+'/',location.href).href,precision=$('precision').value,nextKey=baseUrl+precision;
  if(!engine||nextKey!==key){engine?.destroy();engine=null;key=null;$('progress').hidden=false;
   engine=await createDyna1Engine({baseUrl,precision,signal:controller.signal,cache:'cache',onProgress:p=>{if(p.phase==='weights'){$('progress').value=p.loaded/p.total;$('status').textContent=`Loading weights: ${(p.loaded/1e6).toFixed(0)} / ${(p.total/1e6).toFixed(0)} MB`;}}});key=nextKey;
  }
  $('progress').hidden=true;$('status').textContent='Predicting…';
  result=await engine.predict($('sequence').value,{signal:controller.signal});
  $('status').textContent=`${result.residues.length} residues · ${result.timingMs.toFixed(1)} ms · ${precision.toUpperCase()}`;
  const points=Array.from(result.probs,(p,i)=>`${40+i/Math.max(1,result.probs.length-1)*840},${210-p*190}`).join(' ');
  $('plot').innerHTML=`<path d="M40 20V210H880" fill="none" stroke="#aabeb4"/><text x="12" y="25" font-size="12">1</text><text x="12" y="214" font-size="12">0</text><polyline points="${points}" fill="none" stroke="#175e49" stroke-width="2"/><text x="40" y="234" font-size="12">Residue 1</text><text x="820" y="234" font-size="12">${result.residues.length}</text>`;
  $('residues').replaceChildren(...Array.from(result.residues,(aa,i)=>{const el=document.createElement('span'),p=result.probs[i];el.textContent=aa;el.title=`${i+1} ${aa}: ${p.toFixed(5)}`;el.style.background=`hsl(155 40% ${95-p*60}%)`;el.style.color=p>.6?'white':'#12312c';return el;}));$('download').disabled=false;
 }catch(e){$('status').textContent=e.name==='AbortError'?'Cancelled.':e.message;}
 finally{$('run').disabled=false;$('cancel').disabled=true;$('release').disabled=!engine;$('progress').hidden=true;}
};
$('download').onclick=()=>{if(!result)return;const csv='position,residue,p_exchange,logit\n'+Array.from(result.residues,(r,i)=>`${i+1},${r},${result.probs[i]},${result.logits[i]}`).join('\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv'})),a=document.createElement('a');a.href=url;a.download='dyna1.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
addEventListener('pagehide',()=>engine?.destroy());
