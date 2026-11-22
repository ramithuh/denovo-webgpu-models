import { createProteinaEngine } from "./index.js";
let engine;
const controllers = new Map();
self.addEventListener("message", async ({ data }) => {
  const { id, cmd, args = {} } = data;
  if (cmd === "cancel") {
    controllers.get(args.id)?.abort();
    return;
  }
  const controller = new AbortController();
  controllers.set(id, controller);
  const onProgress = (progress) => self.postMessage({ id, progress });
  try {
    let result;
    if (cmd === "init") {
      if (engine) {
        engine.destroy();
        engine = null;
      }
      engine = await createProteinaEngine({
        ...args,
        signal: controller.signal,
        onProgress,
      });
      result = {
        adapterInfo: engine.adapterInfo,
        isFallbackAdapter: engine.isFallbackAdapter,
        memory: engine.memoryUsage(),
      };
    } else if (cmd === "destroy") {
      engine?.destroy();
      result = engine?.memoryUsage() ?? { totalBytes: 0 };
      engine = null;
    } else {
      if (!engine) throw Error("Initialize the worker first");
      if (cmd === "forward")
        result = await engine.forward(args.coordinates, {
          ...args,
          signal: controller.signal,
        });
      else if (cmd === "sample")
        result = await engine.sample({
          ...args,
          signal: controller.signal,
          onProgress,
        });
      else if (cmd === "memory") result = engine.memoryUsage();
      else throw Error("Unknown worker command");
    }
    const transfer = result?.coordinates ? [result.coordinates.buffer] : [];
    self.postMessage({ id, result }, transfer);
  } catch (e) {
    self.postMessage({ id, error: e.message, name: e.name });
  } finally {
    controllers.delete(id);
  }
});
