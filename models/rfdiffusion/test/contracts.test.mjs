import test from "node:test";
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { browserTask } from "./browser.mjs";
import { root } from "./helpers.mjs";
test(
  "module worker, offline Cache API/OPFS, and storage-denied fallback",
  { timeout: 3600000 },
  async () => {
    const result = await browserTask((page) =>
      page.evaluate(async () => {
        const { loadBundle } = await import("/shared/loader.js"),
          { readFixture, check } = await import("/test/parity.js"),
          read = async (u) => {
            const r = await fetch(u);
            if (!r.ok) throw Error("Missing " + u);
            return r.arrayBuffer();
          };
        const modes = [];
        for (const cache of ["cache", "opfs"]) {
          await loadBundle({ baseUrl: "/weights/base/int8", cache });
          let reads = 0;
          const bundle = await loadBundle({
            baseUrl: "/weights/base/int8",
            cache,
            read: () => {
              reads++;
              throw Error("Network forbidden");
            },
          });
          if (reads || Object.keys(bundle.weights).length !== 5998)
            throw Error(`${cache} offline miss`);
          modes.push(cache);
        }
        const original = navigator.storage.getDirectory;
        try {
          navigator.storage.getDirectory = async () => {
            throw new DOMException("Private mode", "SecurityError");
          };
          let reads = 0;
          await loadBundle({
            baseUrl: "/weights/base/int8",
            cache: "opfs",
            read: async (url) => {
              reads++;
              return read(url);
            },
          });
          if (reads !== 4)
            throw Error("Storage denial did not fetch manifest and 3 shards");
        } finally {
          navigator.storage.getDirectory = original;
        }
        const { createRFdiffusionEngine } = await import("/webgpu/index.js");
        const { requestHardwareAdapter } = await import("/webgpu/device.js");
        const adapter = await requestHardwareAdapter();
        const device = await adapter.requestDevice();
        try {
          const external = await createRFdiffusionEngine({
            baseUrl: "/weights",
            precision: "int8",
            device,
            cache: "none",
          });
          try {
            const f = await readFixture("int8_unconditional_L24", read);
            check(
              (await external.denoiseStep(f.input)).px0,
              f.expected.px0,
              "external device/px0",
            );
          } finally {
            external.destroy();
          }
          if (external.memoryUsage().bytes !== 0)
            throw Error("External engine leaked buffers");
          device.pushErrorScope("validation");
          const probe = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.COPY_DST,
          });
          device.queue.writeBuffer(probe, 0, new Uint32Array([1, 2, 3, 4]));
          await device.queue.onSubmittedWorkDone();
          probe.destroy();
          const error = await device.popErrorScope();
          if (error)
            throw Error("Engine destroyed external device: " + error.message);
        } finally {
          device.destroy();
        }
        const worker = new Worker("/webgpu/worker.js", { type: "module" });
        let id = 0;
        const pending = new Map();
        worker.onmessage = ({ data }) => {
          if (data.progress) return;
          const task = pending.get(data.id);
          if (!task) return;
          pending.delete(data.id);
          data.error
            ? task.reject(Error(data.error.message))
            : task.resolve(data.result);
        };
        worker.onerror = (e) => {
          for (const p of pending.values()) p.reject(Error(e.message));
          pending.clear();
        };
        const call = (method, args = {}) =>
          new Promise((resolve, reject) => {
            pending.set(++id, { resolve, reject });
            worker.postMessage({ id, method, ...args });
          });
        try {
          await call("create", {
            options: { baseUrl: "/weights", precision: "int8" },
          });
          const f = await readFixture("int8_unconditional_L24", read),
            out = await call("denoiseStep", { state: f.input });
          const px0 = check(out.px0, f.expected.px0, "worker/px0");
          if ((await call("memoryUsage")).transientBuffers !== 0)
            throw Error("Worker transient leak");
          if ((await call("destroy")).bytes !== 0)
            throw Error("Worker destruction leak");
          return {
            offline: modes,
            safariStorageDeniedFallback: true,
            externalDeviceOwnership: true,
            workerPx0: px0,
          };
        } finally {
          worker.terminate();
        }
      }),
    );
    assert.deepEqual(result.offline, ["cache", "opfs"]);
    await writeFile(
      root + "fixtures/browser_metrics.json",
      JSON.stringify(result, null, 2) + "\n",
    );
  },
);
