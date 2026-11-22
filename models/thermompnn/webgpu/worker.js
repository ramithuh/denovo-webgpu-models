import { createThermoMPNNEngine } from "./index.js";
let engine = null,
  controller = null,
  busy = false;
function transferables(value, out = new Set()) {
  if (ArrayBuffer.isView(value)) out.add(value.buffer);
  else if (value && typeof value === "object")
    for (const v of Object.values(value)) transferables(v, out);
  return [...out];
}
self.onmessage = async ({ data: { id, command, options, text } }) => {
  if (command === "cancel") {
    controller?.abort();
    return;
  }
  if (busy) {
    self.postMessage({ id, error: "Worker is busy" });
    return;
  }
  busy = true;
  controller = new AbortController();
  try {
    const progress = (event) => self.postMessage({ id, progress: event });
    let result;
    if (command === "init") {
      engine?.destroy();
      engine = null;
      engine = await createThermoMPNNEngine({
        ...options,
        signal: controller.signal,
        onProgress: progress,
      });
      result = {
        precision: engine.precision,
        useF16: engine.useF16,
        memory: engine.memoryUsage(),
      };
    } else if (command === "predict") {
      if (!engine) throw new Error("Initialize the engine first");
      result = await engine.predict(text, {
        ...options,
        signal: controller.signal,
        onProgress: progress,
      });
      result.memory = engine.memoryUsage();
    } else if (command === "destroy") {
      engine?.destroy();
      result = { memory: engine?.memoryUsage() ?? { totalBytes: 0 } };
      engine = null;
    } else throw new Error("Unknown worker command");
    self.postMessage({ id, result }, transferables(result));
  } catch (error) {
    self.postMessage({
      id,
      error: error.message,
      name: error.name,
      memory: engine?.memoryUsage(),
    });
  } finally {
    busy = false;
    controller = null;
  }
};
self.postMessage({ ready: true });
