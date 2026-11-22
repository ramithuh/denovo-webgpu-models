import { createSidProteinEngine } from "./index.js";
let engine, active;
self.onmessage = async ({ data }) => {
  const { id, method, args } = data;
  if (method === "abort") {
    active?.abort();
    return;
  }
  if (active) {
    self.postMessage({ id, error: { name: "Error", message: "Worker busy" } });
    return;
  }
  active = new AbortController();
  try {
    let value;
    if (method === "init") {
      if (engine) throw new Error("Worker already initialized");
      engine = await createSidProteinEngine({
        ...args,
        signal: active.signal,
        onProgress: (event) =>
          self.postMessage({ id, event: "progress", value: event }),
      });
      value = {
        adapterInfo: engine.adapterInfo,
        isFallbackAdapter: engine.isFallbackAdapter,
        memory: engine.memoryUsage(),
      };
    } else if (method === "forward") {
      if (!engine) throw new Error("Worker not initialized");
      value = await engine.forward(args.coordinates, args.time, {
        ...args.options,
        signal: active.signal,
      });
    } else if (method === "generate") {
      if (!engine) throw new Error("Worker not initialized");
      value = await engine.generate({
        ...args,
        signal: active.signal,
        onStep: (event) =>
          self.postMessage({ id, event: "step", value: event }),
      });
    } else if (method === "memoryUsage") {
      value = engine?.memoryUsage() ?? { totalBytes: 0, transientBuffers: 0 };
    } else if (method === "destroy") {
      engine?.destroy();
      value = engine?.memoryUsage() ?? { totalBytes: 0, transientBuffers: 0 };
      engine = null;
    } else throw new Error("Unknown worker method");
    self.postMessage({ id, value });
  } catch (e) {
    self.postMessage({
      id,
      error: { name: e.name, message: e.message, stack: e.stack },
    });
  } finally {
    active = null;
  }
};
