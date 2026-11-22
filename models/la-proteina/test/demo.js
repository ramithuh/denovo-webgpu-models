import {assert} from './parity.js';
const wait=async predicate=>{
  const start=performance.now();while(!predicate()){if(performance.now()-start>120000)throw Error('Demo timed out');await new Promise(r=>setTimeout(r,50));}
};
export async function demoTest(){
  const frame=document.createElement('iframe');frame.src='/demo/';document.body.append(frame);
  try{
    await new Promise((resolve,reject)=>{frame.onload=resolve;frame.onerror=reject;});
    const doc=frame.contentDocument,$=id=>doc.getElementById(id);
    $('length').value='32';$('steps').value='1';$('generate').click();
    await wait(()=>$('generate').disabled===false&&$('status').textContent.startsWith('Generated'));
    assert($('sequence').textContent.length===32,'Demo sequence missing');
    assert(!$('download').hidden,'PDB download missing');
    const response=await fetch($('download').href),pdb=await response.text();assert(pdb.includes('ATOM')&&pdb.endsWith('END\n'),'Demo PDB download invalid');
    const pixel=$('structure').getContext('2d').getImageData(0,0,900,360).data;assert(pixel.some(x=>x>0),'Demo structure preview blank');
    $('steps').value='400';$('generate').click();await wait(()=>$('status').textContent.startsWith('Sampling step'));
    $('cancel').click();await wait(()=>$('status').textContent==='Generation cancelled.');
    $('release').click();await wait(()=>$('status').textContent==='GPU buffers released.');
    return {generation:true,sequence:true,pdbDownload:true,preview:true,cancellation:true,release:true};
  }finally{frame.remove();}
}
