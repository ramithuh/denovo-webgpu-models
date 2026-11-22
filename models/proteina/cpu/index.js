import { loadWeights, decodeTensor } from "../shared/weights.js";
import { network, validateInput } from "../shared/network.js";
import { sampleWith } from "../shared/sampling.js";
import { CPUOps } from "./ops.js";
export async function createProteinaCPU({
  baseUrl,
  precision = "fp32",
  ...options
} = {}) {
  const weights = {};
  const { manifest } = await loadWeights(
    `${baseUrl.replace(/\/$/, "")}/${precision}`,
    {
      ...options,
      expectedPrecision: precision,
      onTensor: (n, t) => {
        weights[n] = { shape: t.shape, data: decodeTensor(t) };
      },
    },
  );
  let destroyed = false;
  const ops = new CPUOps(weights, manifest.constants);
  const forward = async (coordinates, options = {}) => {
    if (destroyed) throw Error("Engine destroyed");
    const input = validateInput(coordinates, options),
      traces = {};
    const start = performance.now();
    const y = network(ops, input, {
      trace: options.trace
        ? (n, t) => {
            traces[n] = t.data.slice();
          }
        : undefined,
    });
    options.signal?.throwIfAborted();
    return {
      coordinates: y.data,
      length: input.length,
      units: "nm",
      timingMs: performance.now() - start,
      trace: traces,
    };
  };
  return {
    forward,
    sample: (options) => sampleWith(forward, options),
    manifest,
    destroy() {
      if (destroyed) return;
      destroyed = true;
      for (const n of Object.keys(weights)) delete weights[n];
    },
  };
}
