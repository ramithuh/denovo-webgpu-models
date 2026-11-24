import { loadWeights } from "../shared/weights.js";
import { forward, normalizeInput } from "../shared/model.js";
import { backbone } from "../shared/geometry.js";
import { sample } from "../shared/sampling.js";
import { acquireDevice } from "./device.js";
import { GPUOps } from "./ops.js";
import { integrate, recordEulerStep } from "./integrate.js";
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
        const checkValidation = ops.hasUnvalidatedPipelines;
        if (checkValidation) context.device.pushErrorScope("validation");
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
          if (checkValidation) ops.hasUnvalidatedPipelines = false;
          signal?.throwIfAborted();
          return {
            pred_trans,
            pred_rotmats,
            coords: backbone(pred_trans, pred_rotmats),
          };
        } finally {
          ops.cancel();
          busy = false;
          if (checkValidation) {
            const error = await context.device.popErrorScope();
            if (error) throw Error(error.message);
          }
        }
      },
      async step(input, eulerOptions = null, { signal, onProgress } = {}) {
        if (destroyed) throw Error("Engine destroyed");
        if (busy) throw Error("Concurrent calls require separate engines");
        signal?.throwIfAborted();
        const x = normalizeInput(input);
        busy = true;
        const checkValidation = ops.hasUnvalidatedPipelines;
        if (checkValidation) context.device.pushErrorScope("validation");
        try {
          ops.begin();
          const transIn = eulerOptions
            ? ops.pinnedTensor(x.trans_t, [x.length, 3])
            : null;
          const rotIn = eulerOptions
            ? ops.pinnedTensor(x.rotmats_t, [x.length, 9])
            : null;
          const out = forward(ops, x, config, (p) => {
            onProgress?.(p);
            signal?.throwIfAborted();
          });
          let targets = [out.pred_trans, out.pred_rotmats];
          if (eulerOptions) {
            const { nextTrans, nextRot } = recordEulerStep(
              ops,
              transIn,
              rotIn,
              out.pred_trans,
              out.pred_rotmats,
              eulerOptions,
            );
            targets = [out.pred_trans, out.pred_rotmats, nextTrans, nextRot];
          }
          const arrays = await ops.finish(targets);
          if (checkValidation) ops.hasUnvalidatedPipelines = false;
          signal?.throwIfAborted();
          return {
            pred_trans: arrays[0],
            pred_rotmats: arrays[1],
            trans_t: arrays[2] ?? null,
            rotmats_t: arrays[3] ?? null,
          };
        } finally {
          ops.cancel();
          busy = false;
          if (checkValidation) {
            const error = await context.device.popErrorScope();
            if (error) throw Error(error.message);
          }
        }
      },
      async integrate(x, pred, options) {
        if (destroyed) throw Error("Engine destroyed");
        if (busy) throw Error("Engine busy");
        busy = true;
        const checkValidation = ops.hasUnvalidatedPipelines;
        if (checkValidation) context.device.pushErrorScope("validation");
        try {
          const res = await integrate(ops, x, pred, options);
          if (checkValidation) ops.hasUnvalidatedPipelines = false;
          return res;
        } finally {
          busy = false;
          if (checkValidation) {
            const error = await context.device.popErrorScope();
            if (error) throw Error(error.message);
          }
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
