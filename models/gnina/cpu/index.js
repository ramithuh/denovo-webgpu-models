import {loadWeights} from "../shared/loader.js";
import {selectModels} from "../shared/network.js";
import {createScorer} from "../shared/api.js";
import {gridCPU} from "../shared/scoring.js";
import {CPUOps} from "./ops.js";
export * from "../shared/index.js";

export async function createGninaCPUEngine(options={}) {
  const names=selectModels(options.ensemble),{tensors}=await loadWeights(options);
  const ops=new CPUOps(tensors);
  ops.grid=input=>({c:28,d:48,data:gridCPU(input)});
  ops.finish=()=>{};
  return createScorer({begin:()=>ops,destroy(){ops.weights={};},memoryUsage:()=>({residentBytes:0,transientBytes:0})},names,options.signal);
}
