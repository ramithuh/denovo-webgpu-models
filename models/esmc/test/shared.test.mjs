import test from "node:test";
import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import {
  tokenize,
  prepareInput,
  scoreMutations,
  ALPHABET,
  parseSafetensors,
  loadWeights,
  diffStats,
} from "../shared/index.js";
test("Exact upstream tokenization, input errors, and mutation indexing", async () => {
  const fixtures = JSON.parse(
    await readFile(new URL("../fixtures/tokenizer.json", import.meta.url)),
  );
  for (const [seq, expected] of Object.entries(fixtures))
    assert.deepEqual([...tokenize(seq)], expected);
  assert.equal(prepareInput("A<unk><mask>C").residues, "AX_C");
  assert.equal(prepareInput([0, 5, 1, 2, 23, 2]).residues.length, 4);
  assert.throws(() => tokenize(""));
  assert.throws(() => prepareInput("AC", 1));
  assert.throws(() => prepareInput({ coordinates: [] }));
  assert.throws(() => prepareInput("AC", 1022, [0]));
  const matrix = Float32Array.from({ length: 40 }, (_, i) => i);
  assert.equal(
    scoreMutations("AC", matrix, [{ position: 1, from: "C", to: "W" }]),
    20 + ALPHABET.indexOf("W"),
  );
  assert.throws(() =>
    scoreMutations("AC", matrix, [{ position: 1, from: "A", to: "W" }]),
  );
  assert.throws(() =>
    scoreMutations("AC", matrix, [
      { position: 1, from: "C", to: "W" },
      { position: 1, from: "C", to: "Y" },
    ]),
  );
});
test("Every committed precision bundle is verified, capped, complete and has colocated scales", async () => {
  for (const precision of ["fp32", "fp16", "int8"]) {
    const base = new URL(`../weights/esmc-300m/${precision}/`, import.meta.url),
      m = JSON.parse(await readFile(new URL("manifest.json", base)));
    assert.equal(m.total_params, 332997184);
    assert.equal(m.num_tensors, Object.keys(m.tensors).length);
    assert.ok(m.shards.every((s) => s.byteLength <= 25000000));
    let n = 0;
    await loadWeights(base.pathname, {
      read: async (p) => {
        const b = await readFile(p);
        return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
      },
      onTensor: (name, t) => {
        n++;
        if (name === "embed.weight" || t.shape.length === 1)
          assert.equal(t.dtype, "F32");
      },
    });
    assert.equal(n, m.num_tensors);
  }
});
test("Safetensors rejects malformed data and digest corruption", async () => {
  assert.throws(() => parseSafetensors(new ArrayBuffer(8)));
  const base = new URL("../weights/esmc-300m/int8/", import.meta.url);
  await assert.rejects(
    () =>
      loadWeights(base.pathname, {
        read: async (p) => {
          const b = await readFile(p);
          if (p.endsWith(".safetensors")) b[b.length - 1] ^= 1;
          return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
        },
      }),
    /checksum/,
  );
  assert.throws(() => diffStats([NaN], [1]));
});
test("Runtime modules have zero external imports or Node builtins", async () => {
  for (const dir of ["shared", "cpu", "webgpu"])
    for (const f of await readdir(new URL(`../${dir}/`, import.meta.url))) {
      if (!f.endsWith(".js")) continue;
      const s = await readFile(
        new URL(`../${dir}/${f}`, import.meta.url),
        "utf8",
      );
      for (const match of s.matchAll(/(?:from\s*|import\s*)['"]([^'"]+)['"]/g))
        assert.ok(match[1].startsWith("."), match[1]);
    }
});
test("Multipart rows reconstruct exactly and malformed manifests fail closed", async () => {
  const { createHash } = await import("node:crypto");
  function shard(key, values) {
    const header = JSON.stringify({
        [key]: { dtype: "F32", shape: [1, 2], data_offsets: [0, 8] },
      }),
      n = Math.ceil(header.length / 8) * 8,
      b = Buffer.alloc(8 + n + 8, 32);
    b.writeBigUInt64LE(BigInt(n));
    b.write(header, 8);
    values.forEach((v, i) => b.writeFloatLE(v, 8 + n + 4 * i));
    return b;
  }
  const a = shard("a", [1, 2]),
    b = shard("b", [3, 4]),
    files = { "a.safetensors": a, "b.safetensors": b };
  const manifest = {
    format: "esmc-safetensors-v1",
    total_params: 4,
    num_tensors: 1,
    total_bytes: a.length + b.length,
    config: {},
    shards: Object.entries(files).map(([file, v]) => ({
      file,
      bytes: v.length,
      byteLength: v.length,
      sha256: createHash("sha256").update(v).digest("hex"),
    })),
    tensors: {
      w: {
        dtype: "F32",
        shape: [2, 2],
        parts: [
          {
            shard: "a.safetensors",
            key: "a",
            dtype: "F32",
            shape: [1, 2],
            row_start: 0,
          },
          {
            shard: "b.safetensors",
            key: "b",
            dtype: "F32",
            shape: [1, 2],
            row_start: 1,
          },
        ],
      },
    },
  };
  const read = async (url) => {
    const b = url.endsWith("manifest.json")
      ? Buffer.from(JSON.stringify(manifest))
      : files[url.split("/").at(-1)];
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  };
  const { decodeTensor } = await import("../shared/index.js");
  assert.deepEqual(
    [...decodeTensor((await loadWeights("/multipart", { read })).weights.w)],
    [1, 2, 3, 4],
  );
  manifest.tensors.w.parts[1].row_start = 0;
  await assert.rejects(() => loadWeights("/multipart", { read }), /multipart/);
  manifest.tensors.w.parts[1].row_start = 1;
  manifest.shards.pop();
  await assert.rejects(
    () => loadWeights("/multipart", { read }),
    /byte count|multipart/,
  );
});

test("CPU invalid wild-type IDs fail before neural computation", async () => {
  const { EsmcCPU } = await import("../cpu/index.js");
  const engine = new EsmcCPU({
    config: { max_sequence_length: 1022, n_layers: 30, d_model: 960 },
    weights: {},
  });
  assert.throws(
    () => engine.forward("AC", { wildTypeTokens: [0, -1, 23, 2] }),
    /wildTypeTokens/,
  );
  assert.throws(
    () => engine.forward("AC", { wildTypeTokens: [0, 5, 23, 64] }),
    /wildTypeTokens/,
  );
  engine.destroy();
});
