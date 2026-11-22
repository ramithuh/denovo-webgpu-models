import { CpuOps } from "./ligandmpnn/ops.js";
import { fetchWeights } from "../shared/ligandmpnn/weights.js";
import { makeEngine } from "../shared/ligandmpnn/model.js";
export { CpuOps };
export { LigandModel } from "../shared/ligandmpnn/model.js";
export async function createLigandMpnnCpuEngine({
  baseUrl,
  variant = "ligandmpnn_v_32_010_25",
  precision = "fp32",
  weights,
  onProgress,
  signal,
} = {}) {
  const bundle =
    weights ??
    (await fetchWeights(`${baseUrl}/${variant}/${precision}`, {
      signal,
      onProgress,
    }));
  return makeEngine(new CpuOps(), bundle, { signal });
}
