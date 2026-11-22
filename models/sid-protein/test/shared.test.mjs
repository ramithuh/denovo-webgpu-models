import { test } from "node:test";
import assert from "node:assert/strict";
import {
  schedule,
  cathIndices,
  gaussianNoise,
  toPDB,
  parseSafetensors,
  loadWeights,
} from "../shared/index.js";
import { acquireDevice, adapterMetadata } from "../webgpu/device.js";
import { root, read, json } from "./io.mjs";
import { readdir, readFile } from "node:fs/promises";
test("exact pinned logarithmic schedules and one-step path", async () => {
  const oracle = await json("fixtures/oracle_fp32.json");
  for (const c of oracle.cases)
    assert.deepEqual([...schedule(c.steps)], c.times);
  assert.throws(() => schedule(0));
  assert.throws(() => schedule(2, 1));
});
test("CATH indices, supplied vocabulary, deterministic Gaussian draws and PDB units", () => {
  assert.deepEqual(cathIndices(), [5, 43, 1472]);
  assert.deepEqual(
    cathIndices({
      cathCode: "1.2.3.4",
      cathMapping: { C: { 1: 0 }, A: { 1.2: 1 }, T: { "1.2.3": 2 } },
    }),
    [0, 1, 2],
  );
  assert.throws(() => cathIndices({ cathCode: "1.2.3" }));
  assert.throws(() => cathIndices({ cathIndices: [6, 0, 0] }));
  assert.deepEqual(gaussianNoise(7, 4), gaussianNoise(7, 4));
  const line = toPDB(Float32Array.of(1, 2, 3)).split("\n")[0];
  assert.equal(line.slice(30, 54), "  10.000  20.000  30.000");
});
test("malformed safetensors and missing bundles fail", async () => {
  assert.throws(() => parseSafetensors(new ArrayBuffer(12)));
  await assert.rejects(loadWeights("missing", { read }));
});
test("adapter retries, software policy and external-device ownership", async () => {
  const real = globalThis.navigator;
  let calls = 0;
  const fakeDevice = {
      features: new Set(),
      adapterInfo: { description: "NVIDIA test hardware" },
    },
    adapter = {
      features: new Set(),
      limits: {},
      info: { description: "SwiftShader" },
      requestDevice: async () => fakeDevice,
    };
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {
      gpu: {
        requestAdapter: async (options) => {
          calls++;
          return options ? null : adapter;
        },
      },
    },
  });
  try {
    const result = await acquireDevice();
    assert.equal(calls, 3);
    assert.equal(result.isFallbackAdapter, true);
    await assert.rejects(
      acquireDevice({ allowSoftwareFallback: false }),
      /Hardware/,
    );
    assert.equal(
      (
        await acquireDevice({
          device: fakeDevice,
          allowSoftwareFallback: false,
        })
      ).ownsDevice,
      false,
    );
    for (const name of ["SwiftShader", "llvmpipe", "WARP"])
      assert.equal(
        adapterMetadata(null, { description: name }).isFallbackAdapter,
        true,
      );
  } finally {
    Object.defineProperty(globalThis, "navigator", {
      configurable: true,
      value: real,
    });
  }
});
test("standalone runtime contains no Node built-ins or external imports", async () => {
  for (const dir of ["shared", "cpu", "webgpu"])
    for (const file of await readdir(new URL(dir + "/", root))) {
      if (!file.endsWith(".js")) continue;
      const text = await readFile(new URL(dir + "/" + file, root), "utf8");
      for (const m of text.matchAll(/(?:from\s*|import\s*\()(['"])(.*?)\1/g))
        assert.match(m[2], /^\./, file + ": external runtime import");
      assert.doesNotMatch(text, /\b(?:require\(|process\.|Buffer\.)/);
    }
});
