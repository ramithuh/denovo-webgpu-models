import test from "node:test";
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { browserTask, root } from "./browser.mjs";

test(
  "complete hardware WebGPU matrix, warm timings, ownership, abort and error cleanup",
  { timeout: 600000 },
  async () => {
    const matrix = await browserTask(async (page, url) =>
      page.evaluate(async (root) => {
        const { hardwareDevice, createBoltzGenEngine } = await import(
          "/webgpu/index.js"
        );
        const {
          configurations,
          cases,
          casesFor,
          paths,
          verifyResult,
          verifyGenerated,
        } = await import("/test/verify.js");
        const hardware = await hardwareDevice(),
          records = [];
        const read = async (url) => {
          const r = await fetch(url);
          if (!r.ok) throw Error(`Missing fixture: ${url}`);
          return r.arrayBuffer();
        };
        try {
          for (const [variant, precision] of configurations) {
            const engine = await createBoltzGenEngine({
              baseUrl: `${root}/weights/${variant}/${precision}`,
              precision,
              device: hardware.device,
            });
            try {
              const memory = engine.memoryUsage();
              if (
                precision === "fp16" &&
                memory.persistentBytes > 1_000_000_000
              )
                throw Error("FP16 expanded in VRAM");
              if (precision === "int8" && memory.persistentBytes > 510_000_000)
                throw Error("INT8 expanded in VRAM");
              for (const scenario of casesFor(variant, precision)) {
                const files = paths(root, variant, precision, scenario),
                  start = performance.now(),
                  result = await engine.run(files),
                  stats = await verifyResult(result, files, read);
                const seconds = (performance.now() - start) / 1000;
                if (engine.memoryUsage().transientBytes !== 0)
                  throw Error("Transient buffers leaked");
                const warmStart = performance.now(),
                  warm = await engine.generate({
                    ...scenario,
                    targetPDB: scenario.targetFile
                      ? await (
                          await fetch(root + "/" + scenario.targetFile)
                        ).text()
                      : undefined,
                  });
                await verifyGenerated(warm, files, read);
                const record = {
                  variant,
                  precision,
                  mode: scenario.name,
                  stats,
                  seconds,
                  warmSeconds: (performance.now() - warmStart) / 1000,
                  warmTimings: warm.timings,
                  coldLoadMs: engine.coldLoadMs,
                  adapter: hardware.info,
                  memory: engine.memoryUsage(),
                };
                records.push(record);
                console.log(
                  "PARITY",
                  variant,
                  precision,
                  scenario.name,
                  JSON.stringify(stats.atom_coords),
                );
              }
              const files = paths(root, variant, precision, cases[0]),
                controller = new AbortController();
              let aborted = false;
              try {
                await engine.run({
                  ...files,
                  signal: controller.signal,
                  onProgress: ({ block }) => {
                    if (block >= 4) controller.abort();
                  },
                });
              } catch (e) {
                aborted = e.name === "AbortError";
              }
              if (!aborted || engine.memoryUsage().transientBytes !== 0)
                throw Error("Abort cleanup failed");
              let error = false;
              try {
                await engine.run({
                  ...files,
                  onProgress: ({ block }) => {
                    if (block >= 4)
                      throw Error("Injected failure during inference");
                  },
                });
              } catch (e) {
                error = e.message === "Injected failure during inference";
              }
              if (!error || engine.memoryUsage().transientBytes !== 0)
                throw Error("Error cleanup failed");
              let stopped = false;
              try {
                await engine.generate({
                  L: 8,
                  numSteps: 3,
                  onProgress: () => engine.destroy(),
                });
              } catch (e) {
                stopped = e.name === "AbortError";
              }
              if (!stopped || engine.memoryUsage().totalBytes !== 0)
                throw Error("Destroy during inference failed");
            } finally {
              engine.destroy();
              engine.destroy();
            }
            if (engine.memoryUsage().totalBytes !== 0)
              throw Error("destroy leaked buffers");
            // External device remains usable after destroying the engine.
            const probe = hardware.device.createBuffer({
              size: 4,
              usage: GPUBufferUsage.STORAGE,
            });
            probe.destroy();
          }
        } finally {
          hardware.device.destroy();
        }
        return records;
      }, url),
    );
    assert.equal(matrix.length, 13);
    for (const record of matrix)
      await writeFile(
        `${root}/fixtures/gpu_${record.variant}_${record.precision}_${record.mode}.json`,
        JSON.stringify(record, null, 2) + "\n",
      );
  },
);

test(
  "module worker executes a complete model and cache/OPFS support offline reads",
  { timeout: 240000 },
  async () => {
    const record = await browserTask(async (page, url) =>
      page.evaluate(async (root) => {
        const { createReader, sha256 } = await import("/shared/loader.js");
        const resource =
            root +
            "/fixtures/dynamic_oracle_diverse_fp32_unconditional_l8_s3.json",
          bytes = await (await fetch(resource)).arrayBuffer(),
          expected = {
            byteLength: bytes.byteLength,
            sha256: await sha256(bytes),
          };
        const cachesChecked = [];
        for (const cache of ["cache", "opfs"]) {
          const reader = createReader({ cache });
          await reader(resource, expected);
          const original = globalThis.fetch;
          globalThis.fetch = () => {
            throw Error("Offline network must not be used");
          };
          try {
            await reader(resource, expected);
            cachesChecked.push(cache);
          } finally {
            globalThis.fetch = original;
          }
        }
        const original = caches.open.bind(caches);
        caches.open = async () => {
          throw Error("Private mode storage denied");
        };
        try {
          await createReader({ cache: "cache" })(resource, expected);
        } finally {
          caches.open = original;
        }
        const worker = new Worker("/webgpu/worker.js", { type: "module" });
        let sequence = 0;
        const pending = new Map();
        worker.onmessage = ({ data }) => {
          if (data.progress) return;
          const p = pending.get(data.id);
          pending.delete(data.id);
          if (data.error) p.reject(Error(data.error.message));
          else p.resolve(data.value);
        };
        worker.onerror = (e) => {
          for (const p of pending.values()) p.reject(Error(e.message));
        };
        const request = (method, options) =>
          new Promise((resolve, reject) => {
            const id = ++sequence;
            pending.set(id, { resolve, reject });
            worker.postMessage({ id, method, options });
          });
        try {
          const init = await request("init", {
            baseUrl: root + "/weights/diverse/fp16",
            precision: "fp16",
          });
          const { paths, verifyResult, cases } = await import(
              "/test/verify.js"
            ),
            files = paths(root, "diverse", "fp16", cases[0]);
          const result = await request("run", files),
            stats = await verifyResult(result, files, async (url) => {
              const r = await fetch(url);
              if (!r.ok) throw Error("Missing fixture");
              return r.arrayBuffer();
            });
          const fresh = await request("generate", {
            mode: "unconditional",
            L: 16,
            numSteps: 10,
            seed: 41,
          });
          if (
            fresh.atom_coords.length !== 16 * 14 * 3 ||
            !fresh.atom_coords.every(Number.isFinite)
          )
            throw Error("Standalone generation failed");
          const extended = [];
          for (const [L, numSteps] of [
            [32, 20],
            [64, 50],
          ]) {
            const start = performance.now(),
              value = await request("generate", { L, numSteps, seed: 47 });
            if (
              value.nAtoms !== L * 14 ||
              value.trajectory.length !== numSteps + 2 ||
              value.denoised.length !== numSteps ||
              !value.atom_coords.every(Number.isFinite)
            )
              throw Error("Dynamic generation contract failed");
            extended.push({
              L,
              numSteps,
              seconds: (performance.now() - start) / 1000,
            });
          }
          const memory = await request("destroy");
          if (memory.totalBytes !== 0) throw Error("Worker leaked buffers");
          return {
            stats,
            adapter: init.adapterInfo,
            cachesChecked,
            privateModeFallback: true,
            newSeedGeneration: true,
            extended,
          };
        } finally {
          worker.terminate();
        }
      }, url),
    );
    assert.equal(record.cachesChecked.length, 2);
    await writeFile(
      `${root}/fixtures/browser_contracts.json`,
      JSON.stringify(record, null, 2) + "\n",
    );
  },
);
