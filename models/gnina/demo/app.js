import {createGninaEngine,parseReceptor,parseLigands,centroid,transformPose} from "../webgpu/index.js";
const el=id=>document.getElementById(id);
const controls=["score","complex","rec","lig","precision"];
let engine,controller,rec,poses,scores,revision=0,angle=.35,tilt=-.26;
function draw(){
  if(!poses) return;
  const canvas=el("pocket"),ctx=canvas.getContext("2d"),rect=canvas.getBoundingClientRect(),scale=devicePixelRatio;
  canvas.width=Math.round(rect.width*scale);canvas.height=Math.round(rect.height*scale);ctx.scale(scale,scale);
  const [cx,cy,cz]=centroid(poses[el("pose").selectedIndex]),points=[];
  for(const [mol,isLig] of [[rec,false],[poses[el("pose").selectedIndex],true]]) for(const atom of mol.atoms){
    const x=atom.x-cx,y=atom.y-cy,z=atom.z-cz;
    if(!isLig&&x*x+y*y+z*z>95) continue;
    const h=x*Math.cos(angle)+z*Math.sin(angle),depth=z*Math.cos(angle)-x*Math.sin(angle);
    points.push({x:h,y:y*Math.cos(tilt)-depth*Math.sin(tilt),z:depth*Math.cos(tilt)+y*Math.sin(tilt),isLig});
  }
  points.sort((a,b)=>a.z-b.z);const zoom=Math.min(rect.width,rect.height)/23;
  for(const p of points) {
    const depth=Math.min(1,Math.max(0,(p.z+10)/20));
    ctx.fillStyle=p.isLig?"#c76323":"#548b7b";ctx.globalAlpha=p.isLig?.98:.14+.34*depth;
    ctx.beginPath();ctx.arc(rect.width/2+p.x*zoom,rect.height/2-p.y*zoom,(p.isLig?.38:.27)*zoom*(.8+.35*depth),0,Math.PI*2);ctx.fill();
  }
}
async function loadInputs(){
  const version=++revision;
  try{
    el("score").disabled=true;el("results").hidden=true;scores=null;
    const id=el("complex").value;
    async function get(role){
      if(id==="custom"){
        const f=el(role).files[0];if(!f) throw Error("Select both a receptor and a ligand file.");
        return [await f.text(),f.name.split(".").pop()];
      }
      const suffix=role==="rec"?"pdb":"sdf",r=await fetch("../fixtures/"+id+"_"+role+"."+suffix);
      if(!r.ok) throw Error("Sample unavailable");return [await r.text(),suffix];
    }
    const [r,l]=await Promise.all([get("rec"),get("lig")]);
    if(version!==revision) return;
    rec=parseReceptor(...r);const [native]=parseLigands(...l);
    poses=[native,transformPose(native,[0,0,Math.PI/2],[9,6,0])];draw();
    el("status").textContent=rec.atoms.length+" receptor atoms · "+native.atoms.length+" ligand atoms";
    el("score").disabled=false;
  }catch(e){if(version===revision){poses=null;el("pocket").getContext("2d").clearRect(0,0,el("pocket").width,el("pocket").height);el("status").textContent=e.message;}}
}
function release(){engine?.destroy();engine=null;el("release").disabled=true;el("status").textContent="GPU released.";}
el("release").onclick=release;el("precision").onchange=release;
el("complex").onchange=loadInputs;
for(const id of ["rec","lig"]) el(id).onchange=()=>{el("complex").value="custom";loadInputs();};
el("pose").onchange=draw;new ResizeObserver(draw).observe(el("pocket"));
el("pocket").onpointerdown=e=>el("pocket").setPointerCapture(e.pointerId);
el("pocket").onpointermove=e=>{if(el("pocket").hasPointerCapture(e.pointerId)){angle+=e.movementX*.01;tilt+=e.movementY*.01;draw();}};
el("cancel").onclick=()=>controller?.abort();
el("score").onclick=async()=>{
  controller=new AbortController();const signal=controller.signal;
  for(const k of controls) el(k).disabled=true;
  el("cancel").disabled=false;el("release").disabled=true;el("results").hidden=true;el("progress").hidden=false;el("progress").removeAttribute("value");
  try{
    el("status").textContent="Loading weights…";
    engine??=await createGninaEngine({baseUrl:"../weights/default",precision:el("precision").value,cache:"cache",signal,
      onProgress:({loaded,total})=>{el("progress").value=loaded/total;}});
    el("status").textContent="Scoring input and perturbed poses…";
    scores=await engine.scorePoses(rec,poses,{signal});
    el("table").replaceChildren();
    for(const [i,s] of scores.entries())for(const [label,row] of [[i?"Perturbed pose":"Input pose",s],...Object.entries(s.models)]){
      const tr=el("table").insertRow();if(row===s) tr.className="ensemble";
      for(const value of [label,...["cnnScore","cnnAffinity","cnnVS","vinaScore"].map(k=>row[k]?.toFixed(5)??"—")]) tr.insertCell().textContent=value;
    }
    el("results").hidden=false;el("status").textContent="Scored with "+el("precision").value.toUpperCase()+".";
  }catch(e){release();el("status").textContent=signal.aborted?"Scoring cancelled.":e.message;}
  finally{for(const k of controls) el(k).disabled=false;el("cancel").disabled=true;el("release").disabled=!engine;el("progress").hidden=true;controller=null;}
};
el("download").onclick=()=>{const a=document.createElement("a");a.download="gnina-scores.json";a.href=URL.createObjectURL(new Blob([JSON.stringify(scores,null,2)],{type:"application/json"}));a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
loadInputs();
