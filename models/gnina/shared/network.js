export const defaultEnsemble=["dense_1.3","dense_1.3_PT_KD_3","crossdock_default2018_KD_4"];
export function selectModels(selection) {
  const names=selection==null||selection==="default"?defaultEnsemble:typeof selection==="string"?[selection]:selection;
  if(!Array.isArray(names)||!names.length||names.some(n=>!defaultEnsemble.includes(n))||new Set(names).size!==names.length) throw Error("Invalid GNINA ensemble");
  return [...names];
}
// Operators own their output. A released activation may be reused by the next dispatch.
export async function forward(ops, grid, name, trace) {
  const layer=s=>name+"/"+s;
  let x=grid;
  function replace(y) {if(x!==grid) ops.release(x);x=y;}
  async function record(key) {if(trace) await trace(layer(key),await ops.read(x));}
  if(name.startsWith("dense")) {
    replace(ops.pool(x,"max",2));
    replace(ops.conv(x,layer("data_enc_init_conv")));
    await record("stem");
    for(let level=0;level<3;level++) {
      const prefix="dense_block_"+level+".data_enc_level"+level;
      for(let i=0;i<4;i++) {
        const norm=ops.norm(x,layer(prefix+"_batchnorm_conv"+i));
        const y=ops.conv(norm,layer(prefix+"_conv"+i));
        ops.release(norm);
        replace(ops.concat(x,y));ops.release(y);
      }
      await record("block"+level);
      if(level<2) {
        replace(ops.conv(x,layer("data_enc_level"+level+"_bottleneck")));
        await record("bottleneck"+level);
        replace(ops.pool(x,"max",2));
      }
    }
    replace(ops.pool(x,"max",x.d));
  } else {
    for(let i=1;i<=5;i++) {
      if(i%2) replace(ops.pool(x,"avg",2));
      replace(ops.conv(x,layer("unit"+i+"_conv")));
      await record("conv"+i);
    }
  }
  const scores=ops.heads(x,layer("pose_output"),layer("affinity_output"));
  const [cnnScore,cnnAffinity]=await ops.read(scores);
  ops.release(scores);ops.release(x);
  if(!Number.isFinite(cnnScore+cnnAffinity)) throw Error("Non-finite neural output");
  return {cnnScore,cnnAffinity,cnnVS:cnnScore*cnnAffinity};
}
