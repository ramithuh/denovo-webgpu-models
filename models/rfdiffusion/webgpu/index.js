import { GPUAlgebra } from "./algebra.js";
import { makeEngine } from "../shared/engine.js";
import { requestHardwareAdapter } from "./device.js";
import { abort } from "../shared/loader.js";
export async function createRFdiffusionEngine(options = {}) {
  const owned = !options.device;
  let device = options.device;
  if (owned) {
    let last;
    for (let attempt = 0; attempt < 3; attempt++) {
      abort(options.signal);
      const adapter = await requestHardwareAdapter();
      try {
        device = await adapter.requestDevice({
          requiredLimits: {
            maxStorageBufferBindingSize:
              adapter.limits.maxStorageBufferBindingSize,
            maxBufferSize: adapter.limits.maxBufferSize,
          },
          requiredFeatures:
            options.useF16 !== false && adapter.features.has("shader-f16")
              ? ["shader-f16"]
              : [],
        });
        break;
      } catch (error) {
        last = error;
        await new Promise((r) => setTimeout(r, 200));
      }
    }
    if (!device) throw last;
  }
  const hasShaderF16 = Boolean(device.features?.has?.("shader-f16"));
  const useF16 = options.useF16 === undefined ? hasShaderF16 : Boolean(options.useF16 && hasShaderF16);
  try {
    const engine = await makeEngine(new GPUAlgebra(device, { useF16 }), options);
    const destroy = engine.destroy;
    let released = false;
    engine.destroy = () => {
      if (released) return;
      released = true;
      destroy();
      if (owned) device.destroy();
    };
    return engine;
  } catch (e) {
    if (owned) device.destroy();
    throw e;
  }
}
