import test from "node:test";
import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import { acquire } from "../webgpu/device.js";
import { toPDB } from "../shared/pdb.js";
import { initialFrames } from "../shared/engine.js";
import { loadBundle } from "../shared/loader.js";
test("production adapter retries and software policy", async () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  let calls = [];
  const fake = {
    info: {
      vendor: "test",
      description: "SwiftShader",
      isFallbackAdapter: true,
    },
    features: new Set(),
    limits: {
      maxStorageBufferBindingSize: 1024,
      maxBufferSize: 1024,
      maxStorageBuffersPerShaderStage: 8,
    },
    requestDevice: async () => ({}),
  };
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {
      gpu: {
        requestAdapter: async (options) => {
          calls.push(options);
          return options ? null : fake;
        },
      },
    },
  });
  try {
    const env = await acquire();
    assert.equal(calls.length, 4);
    assert.equal(env.isFallbackAdapter, true);
    assert.equal(env.ownsDevice, true);
    await assert.rejects(
      () => acquire({ allowSoftwareFallback: false }),
      /Hardware WebGPU required/,
    );
  } finally {
    if (descriptor) Object.defineProperty(globalThis, "navigator", descriptor);
    else delete globalThis.navigator;
  }
});
test("every runtime import stays local and has no Node or npm dependencies", async () => {
  for (const dir of ["shared", "cpu", "webgpu"])
    for (const name of await readdir(
      new URL("../" + dir + "/", import.meta.url),
    )) {
      if (!name.endsWith(".js")) continue;
      const s = await readFile(
        new URL("../" + dir + "/" + name, import.meta.url),
        "utf8",
      );
      for (const m of s.matchAll(/(?:from\s*|import\s*)["']([^"']+)["']/g)) {
        assert.match(m[1], /^\./);
        assert.ok(!m[1].includes("../../"));
      }
      assert.doesNotMatch(s, /node:/);
    }
});
test("deterministic initial frames and complete backbone PDB", () => {
  const a = initialFrames(7, 99),
    b = initialFrames(7, 99);
  assert.deepEqual(a, b);
  for (let i = 0; i < 7; i++)
    assert.ok(Math.abs(Math.hypot(...a.subarray(i * 7, i * 7 + 4)) - 1) < 1e-6);
  const pdb = toPDB({ atoms: new Float32Array(7 * 15), length: 7 });
  assert.equal(pdb.split("\n").filter((l) => l.startsWith("ATOM")).length, 35);
  assert.ok(pdb.endsWith("TER\nEND\n"));
});
test("all export shards and colocated scales are present within size limit", async () => {
  for (const precision of ["fp32", "fp16", "int8"]) {
    const m = JSON.parse(
      await readFile(
        new URL("../weights/" + precision + "/manifest.json", import.meta.url),
      ),
    );
    assert.ok(m.shards.every((s) => s.byteLength <= 25000000));
    for (const e of Object.values(m.tensors)) {
      let row = 0;
      for (const p of e.parts) {
        assert.equal(p.row_start, row);
        row += p.shape[0] ?? 1;
        if (e.dtype === "I8") assert.match(p.scale_tensor, /__scale$/);
      }
      assert.equal(row, e.shape[0] ?? 1);
    }
  }
});
