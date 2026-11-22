// Module-worker RPC. Outputs transfer ownership to the caller without copying.
import { createProtpardelleEngine } from "./index.js";
let engine, controller;
self.onmessage = async ({ data: { id, method, options, input } }) => {
  if (method === "cancel") {
    controller?.abort();
    return;
  }
  try {
    let result;
    if (method === "load") {
      engine?.destroy();
      controller = new AbortController();
      engine = await createProtpardelleEngine({
        ...options,
        signal: controller.signal,
        onProgress: (progress) => self.postMessage({ id, progress }),
      });
      result = { config: engine.config };
    } else if (method === "destroy") {
      engine?.destroy();
      result = { memory: engine?.memoryUsage() };
      engine = null;
    } else {
      if (!engine) throw new Error("Load a model first");
      if (!["denoise", "sequence", "sample", "gradient"].includes(method))
        throw new Error("Unknown worker method");
      controller = new AbortController();
      result = await engine[method](input, {
        ...options,
        signal: controller.signal,
        onProgress: (progress) => self.postMessage({ id, progress }),
      });
      result.memory = engine.memoryUsage();
    }
    const transfers = new Set();
    const visit = (value) => {
      if (ArrayBuffer.isView(value)) transfers.add(value.buffer);
      else if (value && typeof value === "object")
        Object.values(value).forEach(visit);
    };
    visit(result);
    self.postMessage({ id, result }, [...transfers]);
  } catch (error) {
    self.postMessage({
      id,
      error: { name: error.name, message: error.message },
    });
  }
};
