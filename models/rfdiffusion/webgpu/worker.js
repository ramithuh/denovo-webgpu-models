import { createRFdiffusionEngine } from "./index.js";
let engine, controller;
self.onmessage = async ({ data }) => {
  const { id, method, options, state } = data;
  try {
    if (method === "abort") {
      controller?.abort();
      return;
    }
    if (method === "create") {
      engine?.destroy();
      engine = null;
      controller = new AbortController();
      engine = await createRFdiffusionEngine({
        ...options,
        signal: controller.signal,
        onProgress: (p) => self.postMessage({ id, progress: p }),
      });
      self.postMessage({ id, result: { ready: true } });
      return;
    }
    if (!engine) throw Error("Worker engine is not initialized");
    if (method === "destroy") {
      engine.destroy();
      self.postMessage({ id, result: engine.memoryUsage() });
      return;
    }
    if (!["denoiseStep", "sample", "memoryUsage"].includes(method))
      throw Error("Unknown worker method");
    controller = new AbortController();
    const opts = {
        ...options,
        signal: controller.signal,
        onProgress: (p) => self.postMessage({ id, progress: p }),
      },
      result =
        method === "sample"
          ? await engine.sample(opts)
          : method === "denoiseStep"
            ? await engine.denoiseStep(state, opts)
            : engine.memoryUsage();
    self.postMessage({ id, result });
  } catch (error) {
    self.postMessage({
      id,
      error: { name: error.name, message: error.message },
    });
  }
};
