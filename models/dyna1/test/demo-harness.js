export async function run(){
 const frame=document.createElement('iframe');frame.src='/demo/index.html';document.body.append(frame);await new Promise((resolve,reject)=>{frame.onload=resolve;frame.onerror=reject;});
 const doc=frame.contentDocument;doc.getElementById('run').click();
 const start=performance.now();
 while(doc.getElementById('download').disabled){
  if(performance.now()-start>120000)throw new Error('Demo timeout: '+doc.getElementById('status').textContent);
  if(!doc.getElementById('run').disabled)throw new Error('Demo failed: '+doc.getElementById('status').textContent);
  await new Promise(r=>setTimeout(r,100));
 }
 const status=doc.getElementById('status').textContent;
 if(doc.querySelectorAll('#residues span').length!==20||!doc.querySelector('#plot polyline'))throw new Error('Demo results missing');
 doc.getElementById('release').click();if(doc.getElementById('status').textContent!=='Model released.')throw new Error('Demo release failed');
 frame.remove();return {status,residues:20};
}
