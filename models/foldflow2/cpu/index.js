import { loadBundle, toFloat } from "../shared/loader.js";
import { CPUAlgebra } from "./algebra.js";
import { Engine } from "../shared/engine.js";
export async function createFoldFlow2CPU(options = {}) {
  const { manifest, weights } = await loadBundle(options),
    o = new CPUAlgebra(),
    w = {};
  for (const [k, t] of Object.entries(weights))
    w[k] = o.upload(t.shape, toFloat(t), true);
  return new Engine(o, w, manifest);
}
