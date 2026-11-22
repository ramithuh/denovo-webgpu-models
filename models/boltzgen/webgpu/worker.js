import { createBoltzGenEngine } from "./index.js";
let engine,
  initializing = false,
  epoch = 0;
const controllers = new Map();
self.onmessage = async ({ data }) => {
  const { id, method, options = {} } = data;
  if (method === "abort") {
    controllers.get(data.target)?.abort();
    return;
  }
  const controller = new AbortController();
  controllers.set(id, controller);
  try {
    let value;
    if (method === "init") {
      if (initializing)
        throw Error("Worker initialization is already in progress");
      initializing = true;
      const current = ++epoch;
      engine?.destroy();
      engine = null;
      try {
        const loaded = await createBoltzGenEngine({
          ...options,
          signal: controller.signal,
        });
        if (current !== epoch) {
          loaded.destroy();
          throw new DOMException("Initialization cancelled", "AbortError");
        }
        engine = loaded;
        value = {
          adapterInfo: engine.adapterInfo,
          memory: engine.memoryUsage(),
        };
      } finally {
        initializing = false;
      }
    } else if (method === "destroy") {
      epoch++;
      for (const [key, c] of controllers) if (key !== id) c.abort();
      engine?.destroy();
      value = engine?.memoryUsage() ?? {
        totalBytes: 0,
        persistentBytes: 0,
        transientBytes: 0,
      };
      engine = null;
    } else if (method === "generate" || method === "run") {
      if (!engine) throw Error("Initialize worker engine first");
      value = await engine[method]({
        ...options,
        signal: controller.signal,
        onProgress: (progress) => self.postMessage({ id, progress }),
      });
    } else throw Error(`Unknown worker method ${method}`);
    const transfers = new Set();
    const visit = (x) => {
      if (ArrayBuffer.isView(x)) transfers.add(x.buffer);
      else if (x && typeof x === "object") Object.values(x).forEach(visit);
    };
    visit(value);
    self.postMessage({ id, value }, [...transfers]);
  } catch (error) {
    self.postMessage({
      id,
      error: { name: error.name, message: error.message },
    });
  } finally {
    controllers.delete(id);
  }
};
