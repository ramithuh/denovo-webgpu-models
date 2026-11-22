import test from "node:test";
import assert from "node:assert/strict";
import { loadWeights, validateManifest } from "../shared/weights.js";
import { normalizeInput } from "../shared/model.js";
import { acquireDevice } from "../webgpu/device.js";
import {
  linspaceF32,
  randomFrames,
  expSO3,
  logSO3,
  toPDB,
  backbone,
} from "../shared/geometry.js";
import { root, read } from "./helpers.mjs";
test("manifest checksums, packed precisions and strict multipart metadata", async () => {
  for (const precision of ["fp32", "fp16", "int8"]) {
    const b = await loadWeights(root + "weights/" + precision, { read });
    assert.equal(b.manifest.num_tensors, 382);
    for (const s of b.manifest.shards) assert.ok(s.byteLength <= 25000000);
    const bad = structuredClone(b.manifest);
    Object.values(bad.tensors)[0].parts[0].row_start = 1;
    assert.throws(() => validateManifest(bad), /multipart/);
  }
  await assert.rejects(
    loadWeights(root + "weights/fp32", {
      read: async (u) => {
        const b = await read(u);
        if (u.endsWith(".safetensors"))
          new Uint8Array(b)[b.byteLength - 1] ^= 1;
        return b;
      },
    }),
    /checksum/,
  );
  const c = new AbortController();
  c.abort();
  await assert.rejects(
    loadWeights(root + "weights/fp32", { read, signal: c.signal }),
    { name: "AbortError" },
  );
});
test("adapter retries and default production fallback differ from strict tests", async () => {
  const original = Object.getOwnPropertyDescriptor(navigator, "gpu");
  const seen = [];
  let calls = 0;
  const fake = {
    info: {
      vendor: "test",
      description: "SwiftShader",
      isFallbackAdapter: true,
    },
    features: new Set(),
    limits: { maxStorageBufferBindingSize: 1 << 27, maxBufferSize: 1 << 28 },
    async requestDevice() {
      return { destroy() {} };
    },
  };
  Object.defineProperty(navigator, "gpu", {
    configurable: true,
    value: {
      async requestAdapter(o) {
        seen.push(o);
        return ++calls === 1 ? null : fake;
      },
    },
  });
  try {
    const r = await acquireDevice();
    assert.equal(r.isFallbackAdapter, true);
    assert.equal(seen[0].powerPreference, "high-performance");
    assert.equal(seen[1], undefined);
    await assert.rejects(
      acquireDevice({ allowSoftwareFallback: false }),
      /software adapter rejected/,
    );
  } finally {
    if (original) Object.defineProperty(navigator, "gpu", original);
    else delete navigator.gpu;
  }
});
test("SO3 limits, shape validation and PyTorch FP32 time-grid rounding", () => {
  assert.equal(linspaceF32(0.01, 1, 12)[6], 0.5499999523162842);
  for (const v of [
    [0, 0, 0],
    [1e-9, 0, 0],
    [0.2, -0.7, 0.3],
  ]) {
    const round = logSO3(expSO3(v));
    for (let i = 0; i < 3; i++) assert.ok(Math.abs(round[i] - v[i]) < 1e-7);
  }
  const x = randomFrames(17);
  normalizeInput(x);
  assert.throws(() => normalizeInput({ ...x, length: 16 }), /Invalid/);
  assert.throws(
    () => normalizeInput({ ...x, res_mask: new Float32Array(17) }),
    /Masks/,
  );
  assert.equal(
    toPDB(backbone(x.trans_t, x.rotmats_t))
      .split("\n")
      .filter((s) => s.startsWith("ATOM")).length,
    85,
  );
});
