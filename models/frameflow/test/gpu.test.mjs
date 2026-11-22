import test from "node:test";
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { browserTask } from "./browser.mjs";
import { root } from "./helpers.mjs";
test(
  "complete dynamic WGSL network and trajectories on strict hardware",
  { timeout: 3600000 },
  async () => {
    const result = await browserTask((page) =>
      page.evaluate(async () => {
        const liveBuffers = new Map();
        const createBuffer = GPUDevice.prototype.createBuffer;
        const destroyBuffer = GPUBuffer.prototype.destroy;
        GPUDevice.prototype.createBuffer = function (descriptor) {
          const buffer = createBuffer.call(this, descriptor);
          liveBuffers.set(buffer, buffer.size);
          return buffer;
        };
        GPUBuffer.prototype.destroy = function () {
          liveBuffers.delete(this);
          return destroyBuffer.call(this);
        };
        const allocatedBytes = () =>
          [...liveBuffers.values()].reduce((sum, bytes) => sum + bytes, 0);
        const { createFrameFlowEngine } = await import("/webgpu/index.js"),
          { fixture, verify } = await import("/test/parity.js");
        const read = async (url) => {
          const r = await fetch(url);
          if (!r.ok) throw Error("Missing fixture " + url);
          return r.arrayBuffer();
        };
        const matrix = JSON.parse(
            new TextDecoder().decode(await read("/fixtures/matrix.json")),
          ),
          records = [],
          memory = [];
        matrix.cases.push(
          ...JSON.parse(
            new TextDecoder().decode(await read("/fixtures/motif_matrix.json")),
          ).cases,
        );
        let adapter;
        for (const precision of ["fp32", "fp16", "int8"]) {
          const start = performance.now(),
            engine = await createFrameFlowEngine({
              baseUrl: "/weights",
              precision,
              allowSoftwareFallback: false,
            }),
            loadMs = performance.now() - start;
          adapter = engine.adapterInfo;
          try {
            for (const name of matrix.cases.filter((n) =>
              n.startsWith(precision + "_"),
            )) {
              const f = await fixture(name, read, "/");
              records.push({ ...(await verify(engine, f)), loadMs });
              if (engine.memoryUsage().transientBuffers !== 0)
                throw Error("Transient leak");
              if (allocatedBytes() !== engine.memoryUsage().bytes)
                throw Error("Unaccounted live GPU buffers after inference");
              console.log(name);
            }
            const f = await fixture(precision + "_L32_sc0", read, "/");
            records.push({ ...(await verify(engine, f)), warm: true });
            for (const kind of ["abort", "error"]) {
              const controller = new AbortController();
              let caught = false;
              try {
                await engine.forward(f.input, {
                  signal: controller.signal,
                  onProgress() {
                    if (kind === "abort") controller.abort();
                    else throw Error("injected");
                  },
                });
              } catch (e) {
                if (
                  kind === "abort"
                    ? e.name !== "AbortError"
                    : e.message !== "injected"
                )
                  throw e;
                caught = true;
              }
              if (!caught || engine.memoryUsage().transientBuffers !== 0)
                throw Error("Failed " + kind + " cleanup");
              if (allocatedBytes() !== engine.memoryUsage().bytes)
                throw Error("Unaccounted live GPU buffers after " + kind);
            }
            memory.push({ precision, ...engine.memoryUsage() });
          } finally {
            engine.destroy();
            engine.destroy();
          }
          if (engine.memoryUsage().bytes !== 0)
            throw Error("Destroy leaked bytes");
          if (allocatedBytes() !== 0 || liveBuffers.size !== 0)
            throw Error("Actual GPUBuffer destruction incomplete");
        }
        GPUDevice.prototype.createBuffer = createBuffer;
        GPUBuffer.prototype.destroy = destroyBuffer;
        return { adapter, records, memory, actualLiveBuffersAfterDestroy: 0 };
      }),
    );
    assert.equal(result.adapter.isFallbackAdapter, false);
    const b = result.memory.map((x) => x.weightBytes);
    assert.ok(b[1] / b[0] < 0.51);
    assert.ok(b[2] / b[0] < 0.27);
    await writeFile(
      root + "fixtures/gpu_metrics.json",
      JSON.stringify(result, null, 2) + "\n",
    );
  },
);
