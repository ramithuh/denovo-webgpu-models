import { loadWeights } from "../shared/weights.js";
import { Ops } from "../shared/ops.js";
import { forward, normalizeInput } from "../shared/model.js";
import { backbone } from "../shared/geometry.js";
import { sample } from "../shared/sampling.js";
export async function createFrameFlowCPU({
  baseUrl,
  precision = "fp32",
  ...options
}) {
  const { weights, config, manifest } = await loadWeights(
      `${baseUrl.replace(/\/$/, "")}/${precision}`,
      options,
    ),
    ops = new Ops(weights);
  let destroyed = false;
  const engine = {
    manifest,
    async forward(input, { signal, onProgress } = {}) {
      if (destroyed) throw Error("Engine destroyed");
      signal?.throwIfAborted();
      const x = normalizeInput(input),
        out = forward(ops, x, config, (p) => {
          onProgress?.(p);
          signal?.throwIfAborted();
        });
      const result = Object.fromEntries(
        Object.entries(out).map(([k, v]) => [k, v.data]),
      );
      result.coords = backbone(result.pred_trans, result.pred_rotmats);
      return result;
    },
    sample(o) {
      return sample(engine, o);
    },
    destroy() {
      destroyed = true;
      ops.weights = {};
    },
  };
  return engine;
}
