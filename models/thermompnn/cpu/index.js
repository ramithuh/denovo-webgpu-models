import { loadWeights, tensorToFloat, checkAbort } from "../shared/loader.js";
import { parseStructure } from "../shared/structure.js";
import { prepareInput, resultFrom } from "../shared/results.js";
import { executeGraph } from "../shared/graph.js";
import { CPUOps } from "./ops.js";
export async function createThermoMPNNCPU(options = {}) {
  const { manifest, tensors } = await loadWeights(options);
  let weights = Object.fromEntries(
      Object.entries(tensors).map(([k, v]) => [k, tensorToFloat(v)]),
    ),
    busy = false;
  return {
    manifest,
    async predict(text, options = {}) {
      if (!weights) throw new Error("Engine is destroyed");
      if (busy) throw new Error("Inference already in progress");
      checkAbort(options.signal);
      busy = true;
      try {
        const start = performance.now(),
          { structure, mutations } = prepareInput(
            text,
            {
              ...options,
              maxLength: Math.min(options.maxLength ?? 2048, 2048),
            },
            parseStructure,
          );
        const ops = new CPUOps(weights, options),
          ddg = await executeGraph(ops, structure);
        if (options.returnIntermediates) {
          for (const n of ["X", "S", "mask", "chain", "residue_idx"])
            ops.traces[n] = structure[n].slice();
          ops.traces.targeted = Float32Array.from(
            { length: structure.length },
            (_, i) => ddg.data[i * 20],
          );
        }
        return resultFrom(
          structure,
          ddg.data,
          mutations,
          performance.now() - start,
          options.returnIntermediates ? ops.traces : undefined,
        );
      } finally {
        busy = false;
      }
    },
    destroy() {
      weights = null;
    },
    memoryUsage() {
      return {
        residentBytes: weights ? manifest.total_params * 4 : 0,
        transientBuffers: 0,
        transientBytes: 0,
      };
    },
  };
}
