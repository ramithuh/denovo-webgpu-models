import { loadBundle } from "../shared/loader.js";
import { GPUAlgebra } from "./algebra.js";
import { Engine } from "../shared/engine.js";
import { acquire } from "./device.js";
export async function createFoldFlow2Engine(options = {}) {
  const env = await acquire(options),
    o = new GPUAlgebra(env.device);
  try {
    const { manifest, weights } = await loadBundle(options);
    if (options.precision && manifest.precision !== options.precision)
      throw Error("Precision mismatch");
    const w = {};
    for (const [k, t] of Object.entries(weights)) w[k] = o.uploadWeight(t);
    await o.checkpoint([]);
    return Object.assign(new Engine(o, w, manifest), env);
  } catch (e) {
    o.destroy();
    if (env.ownsDevice) env.device.destroy();
    throw e;
  }
}
