import {createLaProteinaWorker} from '../webgpu/worker-client.js';
const $=id=>document.getElementById(id);let engine,loadedPrecision,controller,pdbUrl;
function progress(p){
  if(p.phase==='weights'){$('status').textContent=`Loading weights: ${(p.loaded/1e6).toFixed(0)} / ${(p.total/1e6).toFixed(0)} MB`;$('progress').value=p.loaded/p.total;}
  if(p.phase==='sampling'){$('status').textContent=`Sampling step ${p.step} of ${p.total}`;$('progress').value=p.step/p.total;}
  if(p.phase==='ae1')$('status').textContent=`Decoding all atoms: layer ${p.layer} of ${p.total}`;
}
function draw(ca){
  const canvas=$('structure'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
  const n=ca.length/3,points=[];let lo=[Infinity,Infinity],hi=[-Infinity,-Infinity];
  for(let i=0;i<n;i++){const x=ca[i*3]*.86-ca[i*3+2]*.5,y=ca[i*3+1]*.94+ca[i*3+2]*.34;points.push([x,y]);lo=[Math.min(lo[0],x),Math.min(lo[1],y)];hi=[Math.max(hi[0],x),Math.max(hi[1],y)];}
  const scale=Math.min(820/Math.max(hi[0]-lo[0],.01),300/Math.max(hi[1]-lo[1],.01));
  const xy=p=>[(p[0]-(lo[0]+hi[0])/2)*scale+450,(p[1]-(lo[1]+hi[1])/2)*scale+180];
  ctx.lineWidth=2.5;for(let i=1;i<n;i++){ctx.strokeStyle=`hsl(${150+90*i/n} 55% 65%)`;ctx.beginPath();ctx.moveTo(...xy(points[i-1]));ctx.lineTo(...xy(points[i]));ctx.stroke();}
}
$('controls').addEventListener('submit',async event=>{
  event.preventDefault();controller=new AbortController();$('generate').disabled=true;$('cancel').disabled=false;$('release').disabled=true;
  try{
    const precision=$('precision').value;
    if(engine&&loadedPrecision!==precision){await engine.destroy();engine=null;}
    if(!engine){engine=await createLaProteinaWorker({baseUrl:new URL('../weights',location.href).href,precision,cache:'cache',signal:controller.signal,onProgress:progress});loadedPrecision=precision;}
    $('adapter').textContent=`Adapter: ${engine.adapterInfo?.vendor||'WebGPU'}${engine.isFallbackAdapter?' (software fallback)':''}`;
    const result=await engine.sample({length:Number($('length').value),steps:Number($('steps').value),seed:Number($('seed').value),signal:controller.signal,onProgress:progress});
    draw(result.ca);$('sequence').textContent=result.sequence;
    if(pdbUrl)URL.revokeObjectURL(pdbUrl);pdbUrl=URL.createObjectURL(new Blob([result.pdb],{type:'chemical/x-pdb'}));$('download').href=pdbUrl;$('download').hidden=false;
    $('status').textContent=`Generated ${result.length} residues in ${(result.timingMs/1000).toFixed(2)} seconds.`;$('progress').value=1;
  }catch(error){$('status').textContent=error.name==='AbortError'?'Generation cancelled.':error.message;}
  finally{$('generate').disabled=false;$('cancel').disabled=true;$('release').disabled=!engine;}
});
$('cancel').onclick=()=>controller?.abort();
$('release').onclick=async()=>{if(engine)await engine.destroy();engine=null;$('release').disabled=true;$('status').textContent='GPU buffers released.';};
window.addEventListener('pagehide',()=>{engine?.destroy();if(pdbUrl)URL.revokeObjectURL(pdbUrl);});
