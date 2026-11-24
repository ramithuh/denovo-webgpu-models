import { GPUBackend } from "./backend.js";
import { loadEngine } from "../shared/engine.js";
export async function hardwareDevice() {
  if (!navigator.gpu) throw Error("Hardware WebGPU required");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) throw Error("Hardware WebGPU adapter required");
  const info = adapter.info;
  if (
    adapter.isFallbackAdapter ||
    /swiftshader|llvmpipe|lavapipe|software|subzero/i.test(
      [info.vendor, info.architecture, info.device, info.description].join(" "),
    )
  )
    throw Error("Software WebGPU adapters are forbidden");
  const requiredFeatures = adapter.features.has("shader-f16")
    ? ["shader-f16"]
    : [];
  const requiredLimits = {};
  for (const name of [
    "maxStorageBuffersPerShaderStage",
    "maxStorageBufferBindingSize",
    "maxBufferSize",
  ])
    requiredLimits[name] = adapter.limits[name];
  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits,
  });
  return {
    device,
    info: {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description,
    },
  };
}
export async function createBoltzGenEngine(options = {}) {
  const hardware = options.device
    ? { device: options.device, info: { external: true } }
    : await hardwareDevice();
  const engine = await loadEngine(new GPUBackend(hardware.device), {
    ...options,
    disposeDevice: () => {
      if (!options.device) hardware.device.destroy();
    },
  });
  engine.adapterInfo = hardware.info;
  return engine;
}
