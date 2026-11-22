import { createEsmcEngine } from "../webgpu/index.js";
import { loadWeights } from "../shared/index.js";
import { check } from "./parity.js";
export async function run() {
  const results = [];
  for (const cache of ["cache", "opfs"]) {
    const worker = new Worker("/test/worker.js", { type: "module" });
    try {
      const r = await new Promise((resolve, reject) => {
        worker.onmessage = (e) => resolve(e.data);
        worker.onerror = (e) => reject(new Error(e.message));
        worker.postMessage({ cache });
      });
      check(r.ok, r.error);
      results.push(r);
    } finally {
      worker.terminate();
    }
  }
  // Real model fetch + execution with private-mode storage denial.
  for (const [cache, name] of [
    ["cache", "SecurityError"],
    ["opfs", "QuotaExceededError"],
  ]) {
    const target = cache === "cache" ? globalThis : navigator.storage,
      key = cache === "cache" ? "caches" : "getDirectory",
      descriptor = Object.getOwnPropertyDescriptor(target, key);
    const fail = () => {
      throw new DOMException("Deliberate storage denial", name);
    };
    Object.defineProperty(target, key, {
      configurable: true,
      value: cache === "cache" ? { open: fail } : fail,
    });
    let engine;
    const warnings = [];
    try {
      engine = await createEsmcEngine({
        baseUrl: "/weights/esmc-300m",
        precision: "int8",
        cache,
        onProgress: (p) => {
          if (p.phase === "cache-warning") warnings.push(p.message);
        },
      });
      const out = await engine.forward("AC");
      check(out.logits.length === 256, "Storage-denied inference failed");
      check(warnings.length > 0, "Storage fallback was not exercised");
      results.push({
        cache,
        denial: name,
        executed: true,
        warnings: warnings.length,
      });
    } finally {
      engine?.destroy();
      if (descriptor) Object.defineProperty(target, key, descriptor);
      else delete target[key];
    }
  }
  await caches.delete("esmc-v1");
  await (
    await navigator.storage.getDirectory()
  ).removeEntry("esmc-v1", { recursive: true });
  return { results };
}
