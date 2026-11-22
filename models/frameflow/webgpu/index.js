import { loadWeights } from "../shared/weights.js";
import { forward, normalizeInput } from "../shared/model.js";
import { backbone } from "../shared/geometry.js";
import { sample } from "../shared/sampling.js";
import { acquireDevice } from "./device.js";
import { GPUOps } from "./ops.js";
import { integrate } from "./integrate.js";
export async function createFrameFlowEngine({
  baseUrl,
  precision = "fp32",
  device,
  cache = "none",
  useF16 = false,
  allowSoftwareFallback = true,
  signal,
  onProgress,
  read,
} = {}) {
  if (!["fp32", "fp16", "int8"].includes(precision))
    throw new RangeError("Unknown precision");
  const context = await acquireDevice({
      device,
      allowSoftwareFallback,
      useF16,
      signal,
    }),
    ops = new GPUOps(context.device, { useF16 });
  let destroyed = false,
    busy = false;
  try {
    const { manifest, config } = await loadWeights(
      `${baseUrl.replace(/\/$/, "")}/${precision}`,
      {
        cache,
        signal,
        onProgress,
        read,
        onTensor: (n, t) => ops.uploadWeight(n, t),
      },
    );
    const engine = {
      manifest,
      adapterInfo: context.adapterInfo,
      isFallbackAdapter: context.isFallbackAdapter,
      precision,
      async forward(input, { signal, onProgress } = {}) {
        if (destroyed) throw Error("Engine destroyed");
        if (busy) throw Error("Concurrent calls require separate engines");
        signal?.throwIfAborted();
        const x = normalizeInput(input);
        busy = true;
        context.device.pushErrorScope("validation");
        try {
          ops.begin();
          const out = forward(ops, x, config, (p) => {
            onProgress?.(p);
            signal?.throwIfAborted();
          });
          const [pred_trans, pred_rotmats] = await ops.finish([
            out.pred_trans,
            out.pred_rotmats,
          ]);
          signal?.throwIfAborted();
          return {
            pred_trans,
            pred_rotmats,
            coords: backbone(pred_trans, pred_rotmats),
          };
        } finally {
          ops.cancel();
          busy = false;
          const error = await context.device.popErrorScope();
          if (error) throw Error(error.message);
        }
      },
      async integrate(x, pred, options) {
        if (destroyed) throw Error("Engine destroyed");
        if (busy) throw Error("Engine busy");
        busy = true;
        context.device.pushErrorScope("validation");
        try {
          return await integrate(ops, x, pred, options);
        } finally {
          busy = false;
          const error = await context.device.popErrorScope();
          if (error) throw Error(error.message);
        }
      },
      sample(o) {
        return sample(engine, o);
      },
      memoryUsage() {
        return ops.memoryUsage();
      },
      destroy() {
        if (destroyed) return;
        destroyed = true;
        ops.destroy();
        if (context.owned) context.device.destroy();
      },
    };
    return engine;
  } catch (e) {
    ops.destroy();
    if (context.owned) context.device.destroy();
    throw e;
  }
}
export { toPDB, backbone, randomFrames } from "../shared/geometry.js";
