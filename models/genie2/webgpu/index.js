import { GPUAlgebra } from "./algebra.js";
import { acquireDevice } from "./device.js";
import { initializeEngine } from "../shared/engine.js";
export async function createGenie2Engine(options = {}) {
  const acquired = await acquireDevice(options),
    hasShaderF16 = Boolean(acquired.device.features?.has?.("shader-f16")),
    useF16 = options.useF16 === undefined ? hasShaderF16 : Boolean(options.useF16 && hasShaderF16),
    o = new GPUAlgebra(acquired.device, { useF16 });
  return initializeEngine(o, options, {
    backend: "webgpu",
    adapterInfo: acquired.adapterInfo,
    isFallbackAdapter: acquired.isFallbackAdapter,
    useF16,
    release: () => {
      if (acquired.ownsDevice) acquired.device.destroy();
    },
  });
}
