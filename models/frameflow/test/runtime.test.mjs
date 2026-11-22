import test from "node:test";
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { browserTask } from "./browser.mjs";
import { root } from "./helpers.mjs";
test(
  "real hardware worker, cache, OPFS, denied storage and lifecycle",
  { timeout: 180000 },
  async () => {
    const result = await browserTask((page) =>
      page.evaluate(async () => {
        const { createFrameFlowWorker } = await import("/webgpu/worker.js"),
          { fixture, verify } = await import("/test/parity.js"),
          { loadWeights } = await import("/shared/weights.js");
        const read = async (u) => {
          const r = await fetch(u);
          if (!r.ok) throw Error("Missing fixture");
          return r.arrayBuffer();
        };
        const f = await fixture("int8_L32_sc0", read, "/"),
          worker = await createFrameFlowWorker({
            baseUrl: "/weights",
            precision: "int8",
            allowSoftwareFallback: false,
          });
        let workerParity;
        try {
          workerParity = await verify(worker, f);
          if ((await worker.memoryUsage()).transientBuffers)
            throw Error("Worker leak");
          const abort = new AbortController();
          let caught = false;
          try {
            await worker.sample({
              initial: f.input,
              steps: 100,
              signal: abort.signal,
              onProgress: (p) => {
                if (p.step === 1) abort.abort();
              },
            });
          } catch (e) {
            if (e.name !== "AbortError") throw e;
            caught = true;
          }
          if (!caught || (await worker.memoryUsage()).transientBuffers)
            throw Error("Worker abort failure");
        } finally {
          await worker.destroy();
          await worker.destroy();
        }
        const records = [];
        for (const cache of ["cache", "opfs"]) {
          let first = 0,
            second = 0;
          const original = globalThis.fetch;
          globalThis.fetch = (...a) => {
            first++;
            return original(...a);
          };
          try {
            await loadWeights("/weights/int8", { cache });
          } finally {
            globalThis.fetch = original;
          }
          globalThis.fetch = (...a) => {
            second++;
            return original(...a);
          };
          try {
            await loadWeights("/weights/int8", { cache });
          } finally {
            globalThis.fetch = original;
          }
          if (first === 0 || second !== 0)
            throw Error(`${cache}: cold=${first},warm=${second}`);
          records.push({ cache, coldRequests: first, warmRequests: second });
        }
        let warnings = 0;
        const denied = () => {
          throw new DOMException("Storage denied", "SecurityError");
        };
        const oldOpen = caches.open.bind(caches),
          oldDirectory = navigator.storage.getDirectory.bind(navigator.storage);
        caches.open = denied;
        navigator.storage.getDirectory = denied;
        try {
          for (const cache of ["cache", "opfs"])
            await loadWeights("/weights/int8", {
              cache,
              onProgress: (p) => {
                if (p.phase === "cache-warning") warnings++;
              },
            });
        } finally {
          caches.open = oldOpen;
          navigator.storage.getDirectory = oldDirectory;
        }
        if (warnings === 0)
          throw Error("Denied storage fallback was not exercised");
        return { workerParity, records, deniedStorageWarnings: warnings };
      }),
    );
    assert.equal(result.records.length, 2);
    await writeFile(
      root + "fixtures/runtime_metrics.json",
      JSON.stringify(result, null, 2) + "\n",
    );
  },
);
