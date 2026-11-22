import { createFoldFlow2Engine } from "./index.js";
let engine,
  controller,
  busy = false;
self.onmessage = async ({ data }) => {
  const { id, type, options } = data;
  let ownsOperation = false;
  try {
    if (type === "abort") {
      controller?.abort();
      return;
    }
    let result;
    if (["init", "forward", "sample"].includes(type)) {
      if (busy) throw Error("Worker is busy");
      busy = true;
      ownsOperation = true;
      controller = new AbortController();
    }
    if (type === "init") {
      engine?.destroy();
      engine = undefined;
      engine = await createFoldFlow2Engine({
        ...options,
        signal: controller.signal,
        onProgress: (p) => self.postMessage({ id, progress: p }),
      });
      result = {
        adapterInfo: engine.adapterInfo,
        isFallbackAdapter: engine.isFallbackAdapter,
        memory: engine.memoryUsage(),
      };
    } else if (type === "destroy") {
      controller?.abort();
      engine?.destroy();
      result = engine?.memoryUsage() ?? { bytes: 0, transientBuffers: 0 };
    } else if (type === "memory") {
      result = engine?.memoryUsage() ?? { bytes: 0, transientBuffers: 0 };
    } else if (type === "forward" || type === "sample") {
      if (!engine) throw Error("Worker is not initialized");
      result = await engine[type]({
        ...options,
        signal: controller.signal,
        onProgress: (p) => self.postMessage({ id, progress: p }),
      });
    } else throw Error("Unknown worker request");
    self.postMessage({ id, result });
  } catch (e) {
    self.postMessage({
      id,
      error: { name: e.name, message: e.message, stack: e.stack },
    });
  } finally {
    if (ownsOperation) {
      busy = false;
      controller = undefined;
    }
  }
};
