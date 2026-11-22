import test from "node:test";
import assert from "node:assert/strict";
import { browserTask } from "./browser.mjs";
test(
  "external device ownership and optional native packed f16 decoding",
  { timeout: 120000 },
  async () => {
    const result = await browserTask((page) =>
      page.evaluate(async () => {
        const { acquireDevice } = await import("/webgpu/device.js"),
          { createFrameFlowEngine } = await import("/webgpu/index.js"),
          { fixture, verify } = await import("/test/parity.js");
        const context = await acquireDevice({
            allowSoftwareFallback: false,
            useF16: true,
          }),
          device = context.device;
        const read = async (u) => (await fetch(u)).arrayBuffer();
        try {
          const engine = await createFrameFlowEngine({
            baseUrl: "/weights",
            precision: "fp16",
            device,
            allowSoftwareFallback: false,
            useF16: true,
          });
          try {
            await verify(engine, await fixture("fp16_L32_sc0", read, "/"));
          } finally {
            engine.destroy();
          }
          const buffer = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_DST,
          });
          device.queue.writeBuffer(buffer, 0, new Uint32Array([123]));
          await device.queue.onSubmittedWorkDone();
          buffer.destroy();
          return {
            externalAlive: true,
            nativeF16: device.features.has("shader-f16"),
          };
        } finally {
          device.destroy();
        }
      }),
    );
    assert.equal(result.externalAlive, true);
    assert.equal(result.nativeF16, true);
  },
);
