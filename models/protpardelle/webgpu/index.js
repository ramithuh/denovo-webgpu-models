import { loadBundle, abort } from "../shared/loader.js";
import { denoise, validateInput } from "../shared/denoiser.js";
import { GPUOps } from "./ops.js";
import { requestHardwareAdapter } from "./device.js";
import { miniMPNN } from "../shared/minimpnn.js";
import { denoiseWithGradient } from "../shared/autodiff.js";
import { sample } from "../shared/sampler.js";
export async function createProtpardelleEngine({
  baseUrl,
  variant = "cc58_epoch416",
  precision = "int8",
  device,
  cache = "none",
  useF16 = false,
  signal,
  onProgress,
  read,
} = {}) {
  if (!["fp32", "fp16", "int8"].includes(precision))
    throw new Error("Unsupported precision");
  const owned = !device;
  let ops,
    destroyed = false,
    busy = false,
    lost;
  try {
    abort(signal);
    if (!device) {
      const adapter = await requestHardwareAdapter();
      device = await adapter.requestDevice({
        requiredFeatures:
          useF16 && adapter.features.has("shader-f16") ? ["shader-f16"] : [],
        requiredLimits: {
          maxStorageBufferBindingSize:
            adapter.limits.maxStorageBufferBindingSize,
          maxBufferSize: adapter.limits.maxBufferSize,
        },
      });
    }
    device.lost.then((info) => {
      lost = info.message || "WebGPU device lost";
    });
    const { manifest, weights } = await loadBundle({
      baseUrl: `${baseUrl.replace(/\/$/, "")}/${variant}/${precision}`,
      cache,
      read,
      signal,
      onProgress,
    });
    ops = new GPUOps(device, weights);
    abort(signal);
    return {
      config: manifest.config,
      async sample(
        input,
        { signal, onProgress, returnIntermediates = false } = {},
      ) {
        if (destroyed) throw new Error("Engine destroyed");
        if (busy) throw new Error("Concurrent inference is unsupported");
        abort(signal);
        busy = true;
        let scope = false;
        const start = performance.now();
        try {
          if (
            input.sidechainMode &&
            !ops.weights["mpnn_model.proj_out.weight"]
          ) {
            const bundle = await loadBundle({
              baseUrl: `${baseUrl.replace(/\/$/, "")}/cc58_epoch97_minimpnn/${precision}`,
              cache,
              read,
              signal,
              onProgress,
            });
            ops.attach(bundle.weights);
          }
          device.pushErrorScope("validation");
          scope = true;
          ops.start();
          ops.traceBuffers = new Set();
          const traces = {};
          const result = await sample(ops, manifest.config, input, {
            signal,
            onProgress,
            capture: (name, t) => {
              if (returnIntermediates) {
                traces[name] = t;
                ops.pinned.add(t.buffer);
                ops.traceBuffers.add(t.buffer);
              }
            },
          });
          const out = await ops.readback({ ...result, ...traces });
          const error = await device.popErrorScope();
          scope = false;
          if (error) throw new Error(error.message);
          abort(signal);
          const { coords, aatype, atomMask, ...intermediates } = out;
          return {
            coords,
            aatype: Int32Array.from(aatype),
            atomMask,
            intermediates,
            timingMs: performance.now() - start,
          };
        } finally {
          if (scope) await device.popErrorScope();
          ops.traceBuffers = null;
          ops.cleanup();
          busy = false;
        }
      },
      async gradient(
        input,
        { target, lossMask, signal, returnIntermediates = false } = {},
      ) {
        if (destroyed) throw new Error("Engine destroyed");
        if (busy) throw new Error("Concurrent inference is unsupported");
        abort(signal);
        validateInput(input, manifest.config);
        busy = true;
        device.pushErrorScope("validation");
        let scope = true;
        const start = performance.now();
        try {
          ops.start();
          const traces = {};
          const result = denoiseWithGradient(
            ops,
            manifest.config,
            input,
            target,
            lossMask,
            (name, value) => {
              if (returnIntermediates) traces[name] = value;
            },
          );
          const out = await ops.readback({ ...result, ...traces });
          const error = await device.popErrorScope();
          scope = false;
          if (error) throw new Error(error.message);
          abort(signal);
          const { coords, gradient, ...intermediates } = out;
          return {
            coords,
            gradient,
            intermediates,
            timingMs: performance.now() - start,
          };
        } finally {
          if (scope) await device.popErrorScope();
          ops.cleanup();
          busy = false;
        }
      },
      async sequence(input, { signal, returnIntermediates = false } = {}) {
        if (destroyed) throw new Error("Engine destroyed");
        if (busy) throw new Error("Concurrent inference is unsupported");
        abort(signal);
        validateInput(input, manifest.config);
        busy = true;
        device.pushErrorScope("validation");
        let scope = true;
        const start = performance.now();
        try {
          ops.start();
          const traces = {};
          const output = miniMPNN(
            ops,
            manifest.config,
            input,
            (name, value) => {
              if (returnIntermediates) {
                traces[name] = value;
                ops.pinned.add(value.buffer);
              }
            },
          );
          const results = await ops.readback({ logprobs: output, ...traces });
          const error = await device.popErrorScope();
          scope = false;
          if (error) throw new Error(error.message);
          abort(signal);
          const { logprobs, ...intermediates } = results;
          return {
            logprobs,
            intermediates,
            timingMs: performance.now() - start,
          };
        } finally {
          if (scope) await device.popErrorScope();
          ops.cleanup();
          busy = false;
        }
      },
      async denoise(input, { signal, returnIntermediates = false } = {}) {
        if (destroyed) throw new Error("Engine destroyed");
        if (lost) throw new Error(lost);
        if (busy) throw new Error("Concurrent inference is unsupported");
        abort(signal);
        validateInput(input, manifest.config);
        busy = true;
        const start = performance.now();
        device.pushErrorScope("validation");
        let errorScope = true;
        try {
          ops.start();
          const traces = {};
          const result = denoise(ops, manifest.config, input, (name, value) => {
            if (returnIntermediates) {
              traces[name] = value;
              ops.pinned.add(value.buffer);
            }
          });
          abort(signal);
          const outputs = await ops.readback({ coords: result, ...traces });
          const validation = await device.popErrorScope();
          errorScope = false;
          if (validation) throw new Error(validation.message);
          abort(signal);
          if (outputs.coords.some((x) => !Number.isFinite(x)))
            throw new Error("Nonfinite model output");
          const { coords, ...intermediates } = outputs;
          return { coords, intermediates, timingMs: performance.now() - start };
        } finally {
          if (errorScope) await device.popErrorScope();
          ops.cleanup();
          busy = false;
        }
      },
      memoryUsage: () => ops.memoryUsage(),
      destroy() {
        if (!destroyed) {
          destroyed = true;
          ops.destroy();
          if (owned) device.destroy();
        }
      },
    };
  } catch (error) {
    ops?.destroy();
    if (owned) device?.destroy();
    throw error;
  }
}
