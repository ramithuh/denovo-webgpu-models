import {acquire,Resources} from "./device.js";
import {GPUOps} from "./ops.js";
import {loadWeights,checkAbort} from "../shared/loader.js";
import {createScorer} from "../shared/api.js";
import {selectModels} from "../shared/network.js";
export * from "../shared/index.js";

export async function createGninaEngine(options={}) {
  checkAbort(options.signal);
  const models=selectModels(options.ensemble);
  const resources=new Resources(await acquire(options));
  try {
    const {tensors}=await loadWeights(options);
    await resources.initialize(tensors,models,options.signal);
    resources.begin=check=>new GPUOps(resources,check);
    return createScorer(resources,models,options.signal);
  } catch(error) {resources.destroy();throw error;}
}
