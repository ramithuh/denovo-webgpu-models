import { loadBundle, toFloat, abort } from "./loader.js";
import { RoseTTAFold } from "./network.js";
import constants from "./constants.js";
import { sample, applySymmetry } from "./sampler.js";
export async function makeEngine(
  ops,
  {
    baseUrl,
    precision = "int8",
    cache = "cache",
    read,
    signal,
    onProgress,
  } = {},
) {
  let destroyed = false,
    busy = false;
  try {
    const bundle = await loadBundle({
        baseUrl: `${baseUrl.replace(/\/$/, "")}/base/${precision}`,
        cache,
        read,
        signal,
        onProgress,
      }),
      weights = {};
    for (const [k, v] of Object.entries(bundle.weights)) {
      weights[k] = ops.uploadWeight
        ? ops.uploadWeight(v)
        : ops.upload(v.shape, toFloat(v), true);
    }
    for (const key of Object.keys(bundle.weights)) delete bundle.weights[key];
    const model = new RoseTTAFold(
      ops,
      weights,
      bundle.manifest.config.model,
      constants,
    );
    return {
      async denoiseStep(state, options = {}) {
        if (destroyed) throw Error("Engine destroyed");
        if (busy)
          throw Error("Concurrent inference on one engine is not supported");
        busy = true;
        const start = performance.now();
        try {
          abort(options.signal);
          const out = await model.forward(state, options);
          if (state.symmetry) {
            const sym = await ops.read(
              applySymmetry(
                ops,
                ops.tensor([state.length, 4, 3], out.coords),
                state.symmetry,
              ),
            );
            out.symmetric = sym;
            if (options.returnIntermediates) out.intermediates.symmetric = sym;
          }
          return { ...out, timingMs: performance.now() - start };
        } finally {
          ops.clear();
          busy = false;
        }
      },
      async sample(options = {}) {
        if (destroyed) throw Error("Engine destroyed");
        if (busy)
          throw Error("Concurrent inference on one engine is not supported");
        busy = true;
        try {
          return await sample(
            ops,
            async (state, opts) => {
              try {
                return await model.forward(state, opts);
              } finally {
                ops.clear();
              }
            },
            options,
          );
        } finally {
          ops.clear();
          busy = false;
        }
      },
      memoryUsage: () => ops.memoryUsage(),
      destroy() {
        if (destroyed) return;
        destroyed = true;
        ops.destroy();
        for (const key of Object.keys(weights)) delete weights[key];
      },
    };
  } catch (error) {
    ops.destroy();
    throw error;
  }
}
