import {parseReceptor,parseLigands,transformPose} from "../shared/index.js";

export function relRMS(a,b) {
  if(a.length!==b.length||!a.length) throw Error("Oracle shape mismatch");
  let sq=0,ref=0;
  for(let i=0;i<a.length;i++){sq+=(a[i]-b[i])**2;ref+=b[i]**2;}
  return Math.sqrt(sq/Math.max(ref,1e-30));
}
export async function verify(engine,precision,readText,readFloat,{traces=true}={}) {
  const refs=JSON.parse(await readText("fixtures/oracle_scores.json")), metrics={};
  for(const id of ["184l","10gs"]) {
    const rec=parseReceptor(await readText("fixtures/"+id+"_rec.pdb"));
    const [native]=parseLigands(await readText("fixtures/"+id+"_lig.sdf"));
    const decoy=transformPose(native,[0,0,Math.PI/2],[9,6,0]);
    for(const [label,lig] of [["native",native],["decoy",decoy]]) {
      const key=id+"_"+label, oracle=JSON.parse(await readText(".cache/fixtures/"+key+".json"));
      const layers={},outputs={};
      function check(table,name,a,b) {
        const error=relRMS(a,b);
        if(!(error<1e-4)) throw Error(key+" "+precision+" "+name+" relRMS="+error);
        table[name]=error;
      }
      async function trace(name,values) {
        const slash=name.lastIndexOf("/");
        const t=name==="grid"?oracle.grid:oracle.models[precision+"/"+name.slice(0,slash)][name.slice(slash+1)];
        check(layers,name,values,await readFloat(".cache/fixtures/"+t.file));
      }
      const result=await engine.scorePose(rec,lig,{trace:traces?trace:undefined});
      const ref=refs[key][precision];
      for(const name of ["ensemble",...engine.ensemble]) for(const head of ["cnnScore","cnnAffinity","cnnVS"]) {
        check(outputs,name+"/"+head,[(name==="ensemble"?result:result.models[name])[head]],[(name==="ensemble"?ref:ref.models[name])[head]]);
      }
      if(traces && Object.keys(layers).length!==1+engine.ensemble.reduce((sum,name)=>sum+Object.keys(oracle.models[precision+"/"+name]).length,0)) throw Error("Missing layer traces");
      if(engine.memoryUsage().transientBytes!==0) throw Error("Leaked transient buffers");
      metrics[key]={grid:layers.grid,maxLayer:Math.max(0,...Object.values(layers)),maxOutput:Math.max(...Object.values(outputs)),outputs,layers,cnnScore:result.cnnScore,cnnAffinity:result.cnnAffinity,cnnVS:result.cnnVS,vinaScore:result.vinaScore};
    }
    if(metrics[id+"_native"].cnnScore-metrics[id+"_decoy"].cnnScore<.5) throw Error("Insufficient native/decoy separation");
  }
  return metrics;
}
