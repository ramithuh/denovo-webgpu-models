import { createLigandMpnnEngine } from "../webgpu/index.js";
let engine,
  key,
  controller,
  busy = false;
self.onmessage = async ({ data }) => {
  if (data.type === "cancel") {
    controller?.abort();
    return;
  }
  if (data.type === "destroy") {
    controller?.abort();
    engine?.destroy();
    engine = null;
    key = null;
    return;
  }
  if (data.type !== "design" || busy) return;
  busy = true;
  controller = new AbortController();
  try {
    const next = JSON.stringify(data.config);
    if (next !== key) {
      engine?.destroy();
      engine = null;
      engine = await createLigandMpnnEngine({
        ...data.config,
        signal: controller.signal,
        onProgress: (progress) => postMessage({ type: "progress", progress }),
      });
      key = next;
    }
    const result = await engine.design(data.structure, {
      ...data.options,
      signal: controller.signal,
    });
    postMessage({ type: "result", result });
  } catch (e) {
    postMessage({
      type: "error",
      message: e.name === "AbortError" ? "Cancelled" : e.message,
    });
  } finally {
    busy = false;
  }
};
