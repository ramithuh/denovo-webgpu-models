import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  alphabet,
  encode,
  decode,
  normalizeTokens,
} from "../shared/tokenizer.js";
import {
  loadWeights,
  parseSafetensors,
  validateManifest,
  createReader,
} from "../shared/weights.js";
import { prepareGeneration } from "../shared/sampling.js";
import { rejectSoftware } from "../webgpu/index.js";
const read = async (url) => {
  const b = await readFile(new URL("../" + url, import.meta.url));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
test("exact alphabet, special IDs and invalid inputs", () => {
  assert.equal(alphabet.length, 31);
  assert.equal(decode(encode(alphabet)), alphabet);
  assert.deepEqual(Array.from(encode("-*#@!")), [26, 27, 28, 29, 30]);
  for (const x of ["a", "?", "/", "A\n"]) assert.throws(() => encode(x));
  for (const x of [[31], [-1], [1.5], [NaN]])
    assert.throws(() => normalizeTokens(x));
});
test("generation validates order, draws and fixed positions", () => {
  const a = prepareGeneration({ fixedSequence: "##AC##", seed: 2 });
  assert.equal(a.tokens[2], 0);
  assert.equal(a.tokens[3], 1);
  assert.deepEqual(a, prepareGeneration({ fixedSequence: "##AC##", seed: 2 }));
  for (const options of [
    { length: 0 },
    { length: 4, order: [0, 1, 2, 2] },
    { length: 4, uniforms: [0] },
    { length: 1, uniforms: [1] },
    { fixedSequence: "ABC", maskPositions: [5] },
  ])
    assert.throws(() => prepareGeneration(options));
  assert.deepEqual(
    Array.from(
      prepareGeneration({ fixedSequence: "ACDE", maskPositions: [1, 3] })
        .tokens,
    ),
    [0, 28, 2, 28],
  );
});
test("software adapters rejected", () => {
  for (const description of ["SwiftShader", "llvmpipe", "lavapipe"])
    assert.throws(() => rejectSoftware({ description }));
  assert.throws(() => rejectSoftware({ isFallbackAdapter: true }));
});
test("all bundle shards verified; quantization colocated", async () => {
  for (const p of ["fp32", "fp16", "int8"]) {
    const b = await loadWeights(`weights/oa_dm_38m/${p}`, {
      read,
      cache: "none",
    });
    assert.equal(Object.keys(b.weights).length, 199);
    assert.equal(b.weights["embedder.embedder.weight"].dtype, "F32");
    assert.equal(b.weights["last_norm.weight"].dtype, "F32");
  }
});
test("missing, corrupt, malformed, aborted inputs fail", async () => {
  await assert.rejects(() => loadWeights("missing", { read }));
  await assert.rejects(
    () =>
      loadWeights("weights/oa_dm_38m/int8", {
        read: async (u) => {
          const a = await read(u);
          if (u.endsWith(".safetensors"))
            new Uint8Array(a)[a.byteLength - 1] ^= 1;
          return a;
        },
      }),
    /checksum/,
  );
  assert.throws(() => parseSafetensors(new ArrayBuffer(8)));
  const m = JSON.parse(
    new TextDecoder().decode(
      await read("weights/oa_dm_38m/int8/manifest.json"),
    ),
  );
  m.shards[0].byteLength = 25000001;
  assert.throws(() => validateManifest(m));
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(() =>
    loadWeights("weights/oa_dm_38m/int8", { read, signal: controller.signal }),
  );
});
test("storage write denial falls back to network", async () => {
  const originalFetch = globalThis.fetch,
    originalCaches = globalThis.caches,
    warnings = [];
  globalThis.fetch = async () => new Response("ok");
  globalThis.caches = {
    open: async () => ({
      match: async () => undefined,
      put: async () => {
        throw new DOMException("Private mode", "QuotaExceededError");
      },
    }),
  };
  try {
    assert.equal(
      new TextDecoder().decode(
        await createReader({
          cache: "cache",
          onProgress: (p) => warnings.push(p),
        })("https://example.test/x"),
      ),
      "ok",
    );
    assert.equal(warnings[0].phase, "cache-warning");
  } finally {
    globalThis.fetch = originalFetch;
    globalThis.caches = originalCaches;
  }
});
