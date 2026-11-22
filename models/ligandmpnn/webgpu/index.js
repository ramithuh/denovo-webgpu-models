import { fetchWeights } from "../shared/ligandmpnn/weights.js";
import { makeEngine } from "../shared/ligandmpnn/model.js";
import { GpuOps } from "./ligandmpnn/ops.js";
export { GpuOps };
export { LigandModel } from "../shared/ligandmpnn/model.js";
export * from "./ligandmpnn/shaders.js";
export async function requestWebGpuDevice({ useF16 = true } = {}) {
  if (!globalThis.navigator?.gpu)
    throw Error("WebGPU is unavailable; use HTTPS or localhost");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) throw Error("No WebGPU adapter");
  const hasF16 = adapter.features.has("shader-f16");
  const device = await adapter.requestDevice({
    requiredFeatures: useF16 && hasF16 ? ["shader-f16"] : [],
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  return { adapter, device, hasF16 };
}
export async function createLigandMpnnEngine({
  baseUrl,
  variant = "ligandmpnn_v_32_010_25",
  precision = "fp32",
  device,
  onProgress,
  signal,
  useF16 = true,
  cache = true,
} = {}) {
  if (!baseUrl) throw Error("baseUrl is required");
  if (!["fp32", "fp16", "int8"].includes(precision))
    throw Error("Unsupported precision");
  signal?.throwIfAborted();
  const ownDevice = !device;
  if (!device) ({ device } = await requestWebGpuDevice({ useF16 }));
  try {
    const bundle = await fetchWeights(`${baseUrl}/${variant}/${precision}`, {
      signal,
      onProgress,
      cache,
    });
    signal?.throwIfAborted();
    const ops = new GpuOps(device, {
        half: precision === "fp16",
        useF16: useF16 && device.features.has("shader-f16"),
      }),
      engine = makeEngine(ops, bundle, { signal }),
      destroy = engine.destroy;
    engine.destroy = () => {
      destroy();
      if (ownDevice) device.destroy();
    };
    engine.capabilities = { shaderF16: device.features.has("shader-f16") };
    return engine;
  } catch (e) {
    if (ownDevice) device.destroy();
    throw e;
  }
}
