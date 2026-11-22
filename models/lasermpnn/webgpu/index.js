import { loadWeights } from "../shared/lasermpnn/weights.js";
import { createDesignEngine } from "../shared/lasermpnn/engine.js";
import { GpuBackend } from "./lasermpnn/backend.js";
export { GpuBackend } from "./lasermpnn/backend.js";
export { OP_SHADERS, gemmSource } from "./lasermpnn/shaders.js";
export async function createLaserMpnnEngine({
  baseUrl,
  variant = "laser-0p1A-nothing-heldout",
  precision = "fp32",
  device,
  onProgress,
  signal,
  shaderF16 = false,
  weights,
  cache = true,
} = {}) {
  if (variant !== "laser-0p1A-nothing-heldout")
    throw Error(`Unsupported variant ${variant}`);
  if (!["fp32", "fp16", "int8"].includes(precision))
    throw Error(`Unsupported precision ${precision}`);
  if (shaderF16 && precision !== "fp16")
    throw Error("shaderF16 requires the fp16 weight bundle");
  let owned = false;
  if (!device) {
    if (!globalThis.navigator?.gpu) throw Error("WebGPU is unavailable");
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) throw Error("No WebGPU adapter");
    device = await adapter.requestDevice({
      requiredFeatures: shaderF16 ? ["shader-f16"] : [],
    });
    owned = true;
  }
  if (shaderF16 && !device.features.has("shader-f16")) {
    if (owned) device.destroy();
    throw Error("shader-f16 was requested but is unavailable");
  }
  try {
    signal?.throwIfAborted();
    const bundle =
      weights ?? (await loadWeights(baseUrl, { signal, onProgress, cache }));
    if (bundle.manifest.precision !== precision)
      throw Error("Manifest precision does not match requested precision");
    const backend = await GpuBackend.create(device, bundle.tensors, {
      shaderF16,
    });
    return createDesignEngine(backend, bundle, {
      signal,
      onProgress,
      onDestroy: () => {
        if (owned) device.destroy();
      },
    });
  } catch (e) {
    if (owned) device.destroy();
    throw e;
  }
}
