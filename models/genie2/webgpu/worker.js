import { createGenie2Engine } from "./index.js";
let engine, controller;
self.onmessage = async ({ data }) => {
  const { id, method, options } = data;
  if (method === "abort") {
    controller?.abort();
    return;
  }
  try {
    let result;
    if (method === "init") {
      engine?.destroy();
      engine = await createGenie2Engine(options);
      result = {
        adapterInfo: engine.adapterInfo,
        isFallbackAdapter: engine.isFallbackAdapter,
        precision: engine.precision,
      };
    } else if (method === "destroy") {
      controller?.abort();
      engine?.destroy();
      engine = null;
      result = null;
    } else if (method === "memoryUsage") {
      if (!engine) throw Error("Worker engine not initialized");
      result = engine.memoryUsage();
    } else if (method === "sample" || method === "forward") {
      if (!engine) throw Error("Worker engine not initialized");
      controller = new AbortController();
      result = await engine[method]({
        ...options,
        signal: controller.signal,
        onStep: (step) => self.postMessage({ id, event: "step", step }),
      });
    } else throw Error("Unknown worker method");
    self.postMessage({ id, result });
  } catch (e) {
    self.postMessage({
      id,
      error: { name: e.name, message: e.message, stack: e.stack },
    });
  }
};
