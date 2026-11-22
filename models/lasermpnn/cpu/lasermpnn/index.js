import { loadWeights } from "../../shared/lasermpnn/weights.js";
import { createDesignEngine } from "../../shared/lasermpnn/engine.js";
import { CpuBackend } from "./backend.js";
export { CpuBackend } from "./backend.js";
export { LaserModel } from "../../shared/lasermpnn/model.js";
export async function createLaserMpnnCpuEngine({
  baseUrl,
  weights,
  ...options
} = {}) {
  const bundle = weights ?? (await loadWeights(baseUrl, options));
  return createDesignEngine(new CpuBackend(bundle.tensors), bundle, options);
}
