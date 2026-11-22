import { createEvoDiffEngine } from "../webgpu/index.js";
let engine;
const network = globalThis.fetch.bind(globalThis);
onmessage = async ({ data: { id, cmd, args } }) => {
  try {
    let result;
    if (cmd === "offline") {
      globalThis.fetch = args.value
        ? async () => {
            throw new Error("Network disabled in worker");
          }
        : network;
      if (args.value) {
        let blocked = false;
        try {
          await fetch("/weights/oa_dm_38m/int8/manifest.json");
        } catch {
          blocked = true;
        }
        if (!blocked) throw new Error("Network not blocked");
      }
      result = true;
    }
    if (cmd === "init" || cmd === "privateMode") {
      const progress = [];
      let restore;
      if (cmd === "privateMode") {
        if (args.cache === "cache") {
          const old = globalThis.caches;
          Object.defineProperty(globalThis, "caches", {
            configurable: true,
            value: {
              open: async () => {
                throw new DOMException("Private Mode", "SecurityError");
              },
            },
          });
          restore = () =>
            Object.defineProperty(globalThis, "caches", {
              configurable: true,
              value: old,
            });
        } else {
          const old = navigator.storage.getDirectory;
          navigator.storage.getDirectory = async () => {
            throw new DOMException("Private Mode", "SecurityError");
          };
          restore = () => {
            navigator.storage.getDirectory = old;
          };
        }
      }
      try {
        engine = await createEvoDiffEngine({
          baseUrl: location.origin + "/weights/oa_dm_38m",
          precision: "int8",
          cache: args.cache,
          onProgress: (p) => progress.push(p),
        });
      } finally {
        restore?.();
      }
      if (
        cmd === "privateMode" &&
        !progress.some((p) => p.phase === "cache-warning")
      )
        throw new Error("Fallback not exercised");
      result = { loaded: true };
    }
    if (cmd === "forward") {
      const out = await engine.forward(args.sequence);
      result = { logits: out.logits, memory: engine.memoryUsage() };
    }
    if (cmd === "destroy") {
      engine.destroy();
      result = engine.memoryUsage();
      engine = null;
    }
    postMessage({ id, result }, result?.logits ? [result.logits.buffer] : []);
  } catch (e) {
    postMessage({ id, error: e.stack || String(e) });
  }
};
