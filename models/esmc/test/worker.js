import { createEsmcEngine } from "../webgpu/index.js";
import { diffStats } from "../shared/index.js";
self.onmessage = async ({ data: { cache } }) => {
  let engine;
  const originalFetch = globalThis.fetch;
  try {
    const options = {
      baseUrl: new URL("/weights/esmc-300m", location.href).href,
      precision: "int8",
      cache,
      useF16: false,
    };
    engine = await createEsmcEngine(options);
    const first = await engine.forward("NLYIQWLKDGGPSSGRPPPS");
    engine.destroy();
    globalThis.fetch = () => {
      throw new Error("Network is blocked");
    };
    engine = await createEsmcEngine(options);
    const second = await engine.forward("NLYIQWLKDGGPSSGRPPPS");
    const stats = diffStats(second.logits, first.logits);
    if (stats.relRms !== 0) throw new Error("Offline result differs");
    if (engine.memoryUsage().transientBuffers !== 0)
      throw new Error("Worker transients leaked");
    engine.destroy();
    postMessage({
      ok: true,
      cache,
      offline: true,
      stats,
      bytesAfterDestroy: engine.memoryUsage().totalBytes,
    });
  } catch (e) {
    postMessage({ ok: false, error: e.stack ?? String(e) });
  } finally {
    globalThis.fetch = originalFetch;
    engine?.destroy();
  }
};
