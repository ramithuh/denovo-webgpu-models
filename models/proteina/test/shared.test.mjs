import test from "node:test";
import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import { toPDB, foldIndices } from "../shared/index.js";
import { schedule, noiseSchedule, sampleWith } from "../shared/sampling.js";
import {
  parseSafetensors,
  validateManifest,
  loadWeights,
  createReader,
} from "../shared/weights.js";
import { acquire, rejectSoftware } from "../webgpu/device.js";
import { parity } from "./parity.js";
const root = new URL("../", import.meta.url),
  read = async (url) => {
    const b = await readFile(new URL(url, root));
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  };
test("all six original schedules and three g(t) variants at dynamic step counts", async () => {
  const cases = JSON.parse(
    await readFile(new URL("../fixtures/sampler.json", import.meta.url)),
  );
  for (const c of cases) {
    const ts = schedule(c.steps, c.mode, c.power);
    parity(ts, c.times, c.mode, 2e-6);
    for (const n of c.noise)
      parity(
        noiseSchedule(ts.subarray(0, c.steps), n.mode, n.power, n.clamp),
        n.values,
        "g(t) " + n.mode,
        1e-5,
      );
  }
});
test("PDB nanometer conversion and explicit CATH vocabulary", () => {
  const pdb = toPDB(new Float32Array([1, 2, 3, -1, 0, 0.5]));
  assert.match(pdb, /  10.000  20.000  30.000/);
  assert.equal(pdb.split("\n").filter((x) => x.startsWith("ATOM")).length, 2);
  assert.throws(() => toPDB([NaN, 0, 0]));
  assert.throws(() => foldIndices("1.1.1.x"));
  assert.deepEqual(
    foldIndices("1.2.3.x", { C: { 1: 0 }, A: { 1.2: 2 }, T: { "1.2.3": 3 } }),
    [0, 2, 3],
  );
});
test("safetensors and manifests fail closed on corruption", async () => {
  const m = JSON.parse(
    await readFile(new URL("../weights/int8/manifest.json", import.meta.url)),
  );
  validateManifest(m);
  assert.throws(() => validateManifest({ ...m, total_params: 1 }));
  assert.throws(() => parseSafetensors(new ArrayBuffer(16)));
  await assert.rejects(
    () =>
      loadWeights("weights/int8", {
        read: async (url) => {
          const data = await read(url);
          if (url.endsWith(".safetensors"))
            new Uint8Array(data)[data.byteLength - 1] ^= 1;
          return data;
        },
        cache: "none",
      }),
    /Checksum/,
  );
  const t = structuredClone(m);
  t.tensors.registers.shape = [768, 10];
  assert.throws(() => validateManifest(t));
});
test("production fallback is allowed by default and hardware tests reject software", async () => {
  for (const description of ["SwiftShader", "llvmpipe", "Microsoft WARP"])
    assert.throws(() => rejectSoftware({ description }));
  const device = {
      adapterInfo: { description: "SwiftShader" },
      features: new Set(),
    },
    result = await acquire({ device });
  assert.equal(result.isFallbackAdapter, true);
  await assert.rejects(() => acquire({ device, allowSoftwareFallback: false }));
});
test("adapter acquisition retries then falls back to plain requestAdapter", async () => {
  const old = Object.getOwnPropertyDescriptor(globalThis, "navigator"),
    calls = [];
  let destroyed = 0;
  const device = {
      features: new Set(),
      destroy() {
        destroyed++;
      },
    },
    adapter = {
      info: { vendor: "nvidia" },
      features: new Set(),
      limits: { maxBufferSize: 1024, maxStorageBufferBindingSize: 1024 },
      requestDevice: async () => device,
    };
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: {
      gpu: {
        requestAdapter: async (options) => {
          calls.push(options);
          return calls.length === 4 ? adapter : null;
        },
      },
    },
  });
  try {
    const result = await acquire({ allowSoftwareFallback: false });
    assert.equal(result.device, device);
    assert.equal(calls.length, 4);
    assert.equal(calls[3], undefined);
  } finally {
    if (old) Object.defineProperty(globalThis, "navigator", old);
    else delete globalThis.navigator;
  }
});
test("Safari-style cache exceptions fall back to fetch", async () => {
  const prior = Object.getOwnPropertyDescriptor(globalThis, "caches"),
    fetchOriginal = globalThis.fetch;
  let hits = 0;
  Object.defineProperty(globalThis, "caches", {
    configurable: true,
    get() {
      throw new DOMException("Private", "SecurityError");
    },
  });
  globalThis.fetch = async () => {
    hits++;
    return new Response("ok");
  };
  try {
    assert.equal(
      new TextDecoder().decode(
        await createReader({ cache: "cache" })("https://example.invalid/data"),
      ),
      "ok",
    );
    assert.equal(hits, 1);
  } finally {
    globalThis.fetch = fetchOriginal;
    if (prior) Object.defineProperty(globalThis, "caches", prior);
    else delete globalThis.caches;
  }
});
test("runtime dependency and shader portability boundary", async () => {
  for (const dir of ["shared", "cpu", "webgpu"])
    for (const name of await readdir(
      new URL("../" + dir + "/", import.meta.url),
    )) {
      if (!name.endsWith(".js")) continue;
      const text = await readFile(
        new URL("../" + dir + "/" + name, import.meta.url),
        "utf8",
      );
      assert.doesNotMatch(text, /from\s+['"](?:node:|[^.'"/])/);
      assert.doesNotMatch(text, /\brequire\s*\(/);
      assert.doesNotMatch(text, /const\s+\w+\s*:\s*array/);
    }
});
test("sampler rejects bad controls and implements same-checkpoint CFG", async () => {
  await assert.rejects(() => sampleWith(() => {}, { length: 4, steps: 0 }));
  await assert.rejects(() =>
    sampleWith(() => {}, { length: 4, guidanceWeight: 2 }),
  );
  let calls = 0;
  const out = await sampleWith(
    async (x) => {
      calls++;
      return { coordinates: new Float32Array(x.length) };
    },
    { length: 4, steps: 3, mode: "vf", fold: [1, 1, 1], guidanceWeight: 2 },
  );
  assert.equal(calls, 6);
  assert.equal(out.coordinates.length, 12);
});

test("400-step upstream simulation preserves the FP32 final-ODE cutoff", async () => {
  const ref = JSON.parse(
    await readFile(new URL("../fixtures/sampler_400.json", import.meta.url)),
  );
  let draws = 0;
  const noise = new Proxy(ref.noise, {
    get(target, key) {
      if (/^\d+$/.test(String(key))) draws++;
      return target[key];
    },
  });
  const out = await sampleWith(
    async (x) => ({ coordinates: new Float32Array(x.length) }),
    {
      length: 4,
      steps: 400,
      schedule: "uniform",
      mode: "sc",
      noiseScale: 0.45,
      gtClamp: 10,
      selfCondition: false,
      noise,
    },
  );
  assert.equal(draws, ref.draw_count);
  parity(out.coordinates, ref.output, "400-step scalar simulation");
});

test("time and feature inputs must remain finite and valid after FP32 conversion", async () => {
  const { validateInput } = await import("../shared/network.js");
  assert.throws(() => validateInput([0, 0, 0], { t: 1 - 1e-12 }));
  assert.throws(() => validateInput([0, 0, 0], { xSC: [1e100, 0, 0] }));
});
