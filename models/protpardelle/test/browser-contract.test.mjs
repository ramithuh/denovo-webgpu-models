import test from "node:test";
import { writeFile } from "node:fs/promises";
import { browserTask } from "./browser.mjs";
import { root } from "./helpers.mjs";
test(
  "module worker, offline storage, denied storage, external ownership and error cleanup",
  { timeout: 180000 },
  async () => {
    const result = await browserTask(async (page, url, context) => {
      await page.evaluate(async () => {
        const { createProtpardelleEngine } = await import("/webgpu/index.js"),
          { requestHardwareAdapter } = await import("/webgpu/device.js");
        const meta = await (
            await fetch("/fixtures/oracle_cc58_epoch416.json")
          ).json(),
          bytes = await (
            await fetch("/fixtures/oracle_cc58_epoch416.bin")
          ).arrayBuffer(),
          entry = meta.cases.find(
            (c) => c.mode === "masked_partial" && c.precision === "int8",
          );
        const map = {
          coords: "noisy_coords",
          noiseLevel: "noise_level",
          seqMask: "seq_mask",
          residueIndex: "residue_index",
          chainIndex: "chain_index",
          hotspotMask: "hotspot_mask",
          selfCond: "struct_self_cond",
          cropCond: "struct_crop_cond",
        };
        const input = Object.fromEntries(
          Object.entries(map).map(([k, v]) => {
            const d = entry.inputs[v];
            return [
              k,
              d
                ? new (d.dtype === "I32" ? Int32Array : Float32Array)(
                    bytes.slice(d.offset, d.offset + d.length * 4),
                  )
                : null,
            ];
          }),
        );
        globalThis.contract = {
          createProtpardelleEngine,
          requestHardwareAdapter,
          input,
          options: {
            baseUrl: location.origin + "/weights",
            variant: "cc58_epoch416",
            precision: "int8",
          },
        };
        const worker = new Worker("/webgpu/worker.js", { type: "module" });
        let counter = 0;
        const rpc = (method, params = {}) =>
          new Promise((resolve, reject) => {
            const id = ++counter;
            const listener = ({ data }) => {
              if (data.id !== id || data.progress) return;
              worker.removeEventListener("message", listener);
              data.error
                ? reject(new Error(data.error.message))
                : resolve(data.result);
            };
            worker.addEventListener("message", listener);
            worker.postMessage({ id, method, ...params });
          });
        try {
          await rpc("load", { options: contract.options });
          const result = await rpc("denoise", { input });
          if (
            result.coords.length !== 888 ||
            result.coords.some((x) => !Number.isFinite(x)) ||
            result.memory.transientBuffers
          )
            throw new Error("Worker inference failed");
          const released = await rpc("destroy");
          if (released.memory.bytes) throw new Error("Worker release failed");
        } finally {
          worker.terminate();
        }
      });
      const storage = [];
      for (const cache of ["cache", "opfs"]) {
        await page.evaluate(async (cache) => {
          const c = contract,
            e = await c.createProtpardelleEngine({ ...c.options, cache });
          try {
            await e.denoise(c.input);
          } finally {
            e.destroy();
          }
        }, cache);
        await context.setOffline(true);
        try {
          await page.evaluate(async (cache) => {
            const c = contract,
              e = await c.createProtpardelleEngine({ ...c.options, cache });
            try {
              await e.denoise(c.input);
            } finally {
              e.destroy();
            }
          }, cache);
          storage.push({ cache, offline: true });
        } finally {
          await context.setOffline(false);
        }
      }
      await page.evaluate(async () => {
        const c = contract;
        for (const [name, cache] of [
          ["SecurityError", "cache"],
          ["QuotaExceededError", "opfs"],
        ]) {
          const object = cache === "cache" ? globalThis : navigator.storage,
            key = cache === "cache" ? "caches" : "getDirectory",
            old = Object.getOwnPropertyDescriptor(object, key);
          Object.defineProperty(object, key, {
            configurable: true,
            value:
              cache === "cache"
                ? {
                    open: async () => {
                      throw new DOMException("Denied", name);
                    },
                  }
                : async () => {
                    throw new DOMException("Denied", name);
                  },
          });
          try {
            const e = await c.createProtpardelleEngine({ ...c.options, cache });
            try {
              await e.denoise(c.input);
            } finally {
              e.destroy();
            }
          } finally {
            if (old) Object.defineProperty(object, key, old);
            else delete object[key];
          }
        }
        const adapter = await c.requestHardwareAdapter(),
          device = await adapter.requestDevice();
        const engine = await c.createProtpardelleEngine({
          ...c.options,
          device,
        });
        try {
          let callbackFailed = false;
          try {
            await engine.sample(
              {
                seqMask: c.input.seqMask,
                residueIndex: c.input.residueIndex,
                chainIndex: c.input.chainIndex,
                numSteps: 2,
              },
              {
                onProgress: () => {
                  throw new Error("progress callback failure");
                },
              },
            );
          } catch (e) {
            callbackFailed = e.message === "progress callback failure";
          }
          if (!callbackFailed || engine.memoryUsage().transientBuffers)
            throw new Error("In-flight error cleanup failed");
          let rejected = false;
          try {
            await engine.denoise({ ...c.input, coords: new Float32Array(2) });
          } catch {
            rejected = true;
          }
          if (!rejected || engine.memoryUsage().transientBuffers)
            throw new Error("Error cleanup failed");
          const controller = new AbortController();
          await engine
            .sample(
              {
                seqMask: c.input.seqMask,
                residueIndex: c.input.residueIndex,
                chainIndex: c.input.chainIndex,
                numSteps: 5,
              },
              {
                signal: controller.signal,
                onProgress: () => controller.abort(),
              },
            )
            .then(
              () => {
                throw new Error("Expected abort");
              },
              (e) => {
                if (e.name !== "AbortError") throw e;
              },
            );
          if (engine.memoryUsage().transientBuffers)
            throw new Error("Sampler cancellation leaked");
        } finally {
          engine.destroy();
          engine.destroy();
        }
        device.pushErrorScope("validation");
        const probe = device.createBuffer({
          size: 4,
          usage: GPUBufferUsage.STORAGE,
        });
        await device.queue.onSubmittedWorkDone();
        const error = await device.popErrorScope();
        probe.destroy();
        device.destroy();
        if (error) throw new Error("Caller device was destroyed");
        const admissionDevice = await (
            await c.requestHardwareAdapter()
          ).requestDevice(),
          { GPUOps } = await import("/webgpu/ops.js"),
          ops = new GPUOps(admissionDevice, {});
        let rejected = false;
        try {
          ops.allocate(admissionDevice.limits.maxStorageBufferBindingSize + 4);
        } catch (e) {
          rejected = /admission/.test(e.message);
        } finally {
          ops.destroy();
          admissionDevice.destroy();
        }
        if (!rejected || ops.memoryUsage().bytes)
          throw new Error("Missing buffer admission check");
      });
      return {
        worker: true,
        storage,
        deniedStorage: true,
        externalDevice: true,
        cancellation: true,
        admission: true,
      };
    });
    await writeFile(
      root + "fixtures/browser_contract_metrics.json",
      JSON.stringify(result, null, 2) + "\n",
    );
  },
);
