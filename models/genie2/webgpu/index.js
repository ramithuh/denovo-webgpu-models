import { GPUAlgebra } from "./algebra.js";
import { acquireDevice } from "./device.js";
import { initializeEngine } from "../shared/engine.js";
export async function createGenie2Engine(options = {}) {
  const acquired = await acquireDevice(options),
    o = new GPUAlgebra(acquired.device);
  return initializeEngine(o, options, {
    backend: "webgpu",
    adapterInfo: acquired.adapterInfo,
    isFallbackAdapter: acquired.isFallbackAdapter,
    useF16: !!options.useF16 && acquired.device.features.has("shader-f16"),
    release: () => {
      if (acquired.ownsDevice) acquired.device.destroy();
    },
  });
}
