import test from "node:test";
import assert from "node:assert/strict";
import { acquireDevice, adapterMetadata } from "../webgpu/device.js";
test("Production retries high-performance then plain adapter, permits software by default; strict policy rejects it", async () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  const calls = [],
    device = { destroy() {} };
  const adapter = {
    info: { vendor: "SwiftShader", isFallbackAdapter: false },
    features: new Set(),
    limits: {
      maxBufferSize: 100,
      maxStorageBufferBindingSize: 100,
      maxStorageBuffersPerShaderStage: 8,
    },
    requestDevice: async () => device,
  };
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {
      gpu: {
        requestAdapter: async (arg) => {
          calls.push(arg);
          return arg ? null : adapter;
        },
      },
    },
  });
  try {
    const got = await acquireDevice();
    assert.equal(got.device, device);
    assert.equal(got.isFallbackAdapter, true);
    assert.equal(calls.length, 4);
    assert.deepEqual(
      calls.slice(0, 3),
      Array(3).fill({ powerPreference: "high-performance" }),
    );
    assert.equal(calls[3], undefined);
    await assert.rejects(
      acquireDevice({ allowSoftwareFallback: false }),
      /Hardware WebGPU required/,
    );
  } finally {
    if (original) Object.defineProperty(globalThis, "navigator", original);
    else delete globalThis.navigator;
  }
  for (const vendor of ["SwiftShader", "llvmpipe", "WARP"])
    assert(
      adapterMetadata({ info: { vendor, isFallbackAdapter: false } })
        .isFallbackAdapter,
    );
});
