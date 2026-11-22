import { createFrameFlowEngine } from "./index.js";
let engine;
const controllers = new Map();
self.onmessage = async ({ data }) => {
  const { id, method, args } = data;
  if (method === "abort") {
    controllers.get(data.target)?.abort();
    return;
  }
  const controller = new AbortController();
  controllers.set(id, controller);
  const opts = {
    signal: controller.signal,
    onProgress: (progress) => self.postMessage({ id, progress }),
  };
  try {
    let result;
    if (method === "init") {
      engine?.destroy();
      engine = await createFrameFlowEngine({ ...args, ...opts });
      result = {
        adapterInfo: engine.adapterInfo,
        isFallbackAdapter: engine.isFallbackAdapter,
        precision: engine.precision,
      };
    } else if (!engine) throw Error("Initialize the worker first");
    else if (method === "forward") result = await engine.forward(args, opts);
    else if (method === "sample")
      result = await engine.sample({ ...args, ...opts });
    else if (method === "memory") result = engine.memoryUsage();
    else if (method === "destroy") {
      for (const [key, c] of controllers) if (key !== id) c.abort();
      engine.destroy();
      result = engine.memoryUsage();
    } else throw Error(`Unknown worker operation ${method}`);
    self.postMessage({ id, result });
  } catch (e) {
    self.postMessage({ id, error: { name: e.name, message: e.message } });
  } finally {
    controllers.delete(id);
  }
};
