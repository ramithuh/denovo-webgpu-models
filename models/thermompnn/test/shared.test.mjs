import test from "node:test";
import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import {
  parseStructure,
  cifTokens,
  resolveMutations,
} from "../shared/structure.js";
import {
  loadWeights,
  parseSafetensors,
  tensorToFloat,
} from "../shared/loader.js";
import { bestMutants, centrality, toCSV, toFASTA } from "../shared/results.js";
import { topk } from "../cpu/topk.js";
import { assertHardware } from "../webgpu/index.js";
import { read, readJSON, oracle } from "./helpers.mjs";
const text = async (f) => new TextDecoder().decode(await read("fixtures/" + f));

test("PDB/mmCIF preprocessing matches upstream exactly for every mode", async () => {
  for (const c of await readJSON("fixtures/cases.json")) {
    const s = parseStructure(await text(c.input), {
        chains: c.inputChains ?? c.chains,
      }),
      { traces } = await oracle(c, "fp32");
    for (const key of ["X", "S", "mask", "chain", "residue_idx"])
      assert.deepEqual(s[key], traces[key], `${c.id}/${key}`);
  }
  const s = parseStructure(await text("stress.pdb"));
  assert.equal(s.residues[0].number, -2);
  assert.equal(s.residues[2].aa, "M");
  assert.ok(s.residues.some((r) => r.number === 1 && r.missing));
  assert.ok(s.residues.some((r) => r.number === 3 && r.ins === "A"));
  assert.equal(s.length, 11);
  assert.equal(s.mask.filter((x) => !x).length, 2);
  const cif = parseStructure(await text("complex.cif"));
  assert.deepEqual(cif.chains, ["Alpha", "Beta"]);
  const order = parseStructure(await text("complex.pdb"), {
    chains: ["B", "A"],
  });
  assert.equal(order.residues[0].chain, "B");
  const insertion = resolveMutations(s, [
    { chain: "A", number: 3, ins: "A", mutation: "A" },
  ]);
  assert.equal(insertion[0].ins, "A");
});
test("Tokenizer handles quotes, comments, multiline values; bad inputs fail clearly", async () => {
  assert.deepEqual(
    cifTokens("data_t # comment\n'a b' \"c d\"\n;line one\nline two\n;\n"),
    ["data_t", "a b", "c d", "line one\nline two"],
  );
  for (const x of ["", "HEADER no atoms"])
    assert.throws(() => parseStructure(x));
  const p = await text("small.pdb"),
    s = parseStructure(p);
  assert.throws(() => parseStructure(p, { chains: [] }));
  assert.throws(() => parseStructure(p, { chains: ["missing"] }));
  assert.throws(() => parseStructure(p, { chains: ["A", "A"] }));
  assert.throws(() => parseStructure(p, { maxLength: 1 }));
  assert.throws(() => parseStructure(p, { format: "fasta" }));
  assert.throws(() =>
    parseStructure(
      p.replace("ATOM", "ATOM").slice(0, 30) + "     NaN" + p.slice(38),
    ),
  );
  for (const m of [
    [{ index: -1, mutation: "A" }],
    [{ index: 0, mutation: "AA" }],
    [{ index: 0, mutation: "X" }],
    [{ index: 0, mutation: "A", wildtype: "?" }],
    [],
  ])
    assert.throws(() => resolveMutations(s, m));
  assert.throws(() =>
    resolveMutations(parseStructure(awaitTextStress), [
      { chain: "A", number: 1, mutation: "A" },
    ]),
  );
  assert.throws(() => cifTokens("'broken"));
  assert.throws(() => cifTokens(";broken"));
});
const awaitTextStress = await text("stress.pdb");
function safe(raw) {
  const header = {},
    chunks = [];
  let offset = 0;
  for (const [key, { data, dtype, shape }] of Object.entries(raw)) {
    header[key] = {
      dtype,
      shape,
      data_offsets: [offset, offset + data.byteLength],
    };
    chunks.push(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    offset += data.byteLength;
  }
  let h = JSON.stringify(header);
  h += " ".repeat((8 - (h.length % 8)) % 8);
  const b = new Uint8Array(8 + h.length + offset);
  new DataView(b.buffer).setBigUint64(0, BigInt(h.length), true);
  b.set(new TextEncoder().encode(h), 8);
  offset = 8 + h.length;
  for (const c of chunks) {
    b.set(c, offset);
    offset += c.length;
  }
  return b.buffer;
}
function bundle() {
  const b1 = safe({
    a: { data: new Int8Array([127, -127, 64, 0]), dtype: "I8", shape: [2, 2] },
    "a.__scale": {
      data: new Float32Array([0.5, 0.25]),
      dtype: "F32",
      shape: [2],
    },
  });
  const b2 = safe({
    b: { data: new Int8Array([2, -2]), dtype: "I8", shape: [1, 2] },
    "b.__scale": { data: new Float32Array([2]), dtype: "F32", shape: [1] },
  });
  const buffers = { "one.safetensors": b1, "two.safetensors": b2 };
  const m = {
    format: "thermompnn-safetensors-v1",
    model: "ThermoMPNN",
    precision: "int8",
    total_params: 6,
    num_tensors: 1,
    total_bytes: b1.byteLength + b2.byteLength,
    shards: Object.entries(buffers).map(([file, b]) => ({
      file,
      byteLength: b.byteLength,
      sha256: createHash("sha256").update(new Uint8Array(b)).digest("hex"),
    })),
    tensors: {
      w: {
        shape: [3, 2],
        dtype: "I8",
        parts: [
          {
            key: "a",
            shard: "one.safetensors",
            shape: [2, 2],
            dtype: "I8",
            row_start: 0,
            quant: { scheme: "symmetric_per_row", scale_tensor: "a.__scale" },
          },
          {
            key: "b",
            shard: "two.safetensors",
            shape: [1, 2],
            dtype: "I8",
            row_start: 2,
            quant: { scheme: "symmetric_per_row", scale_tensor: "b.__scale" },
          },
        ],
      },
    },
  };
  return {
    m,
    buffers,
    read: async (url) =>
      url.endsWith("manifest.json")
        ? new TextEncoder().encode(JSON.stringify(m)).buffer
        : buffers[url.split("/").at(-1)],
  };
}
test("Loader reconstructs multipart I8 with colocated scales, strict SHA and metadata checks", async () => {
  const b = bundle(),
    loaded = await loadWeights({
      baseUrl: "test",
      precision: "int8",
      read: b.read,
    });
  assert.deepEqual(
    [...tensorToFloat(loaded.tensors.w)],
    [63.5, -63.5, 16, 0, 4, -4],
  );
  const corrupt = bundle();
  new Uint8Array(corrupt.buffers["one.safetensors"])[90] ^= 1;
  await assert.rejects(
    loadWeights({ baseUrl: "test", precision: "int8", read: corrupt.read }),
    /integrity/,
  );
  const gap = bundle();
  gap.m.tensors.w.parts[1].row_start = 1;
  await assert.rejects(
    loadWeights({ baseUrl: "test", precision: "int8", read: gap.read }),
    /multipart/,
  );
  const scale = bundle();
  scale.m.tensors.w.parts[1].quant.scale_tensor = "a.__scale";
  await assert.rejects(
    loadWeights({ baseUrl: "test", precision: "int8", read: scale.read }),
    /scales/,
  );
  const total = bundle();
  total.m.total_params++;
  await assert.rejects(
    loadWeights({ baseUrl: "test", precision: "int8", read: total.read }),
    /totals/,
  );
  assert.throws(() => parseSafetensors(new ArrayBuffer(1)));
  assert.throws(() => parseSafetensors(new Uint8Array(16).fill(255).buffer));
  await assert.rejects(
    loadWeights({ baseUrl: "weights/default", precision: "nope", read }),
    /precision/,
  );
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    loadWeights({
      baseUrl: "weights/default",
      read,
      signal: controller.signal,
    }),
  );
  await assert.rejects(loadWeights({ baseUrl: "missing", read }));
});
test("Runtime has zero external dependencies or Node imports; software adapters fail", async () => {
  for (const dir of ["shared", "cpu", "webgpu"])
    for (const file of await readdir(
      new URL("../" + dir + "/", import.meta.url),
    )) {
      if (!file.endsWith(".js")) continue;
      const source = await readFile(
        new URL("../" + dir + "/" + file, import.meta.url),
        "utf8",
      );
      for (const m of source.matchAll(
        /(?:from\s*|import\s*\()\s*['"]([^'"]+)['"]/g,
      ))
        assert.ok(m[1].startsWith("."), `${dir}/${file}: ${m[1]}`);
      assert.ok(!/\brequire\s*\(|\bprocess\./.test(source));
    }
  for (const description of [
    "SwiftShader",
    "llvmpipe",
    "lavapipe",
    "Software Adapter",
    "Microsoft Basic Render Driver",
  ])
    assert.throws(() => assertHardware({ info: { description } }));
  assert.throws(() => assertHardware(null));
  assert.throws(() => assertHardware({ isFallbackAdapter: true }));
});
test("Postprocessing, downloads and centrality retain chain/residue identity", async () => {
  const s = parseStructure(await text("stress.pdb")),
    ddg = new Float32Array(s.length * 20);
  ddg.fill(1);
  for (let i = 0; i < s.length; i++) {
    ddg[i * 20] = 0;
    ddg[i * 20 + 1] = -1;
  }
  const result = {
    ...s,
    ddg,
    valid: Uint8Array.from(s.residues, (r) => r.aa !== "X"),
    alphabet: "ACDEFGHIKLMNPQRSTVWY",
  };
  assert.ok(bestMutants(result).every((r) => r.mutation === "C"));
  assert.ok(
    bestMutants(result, { includeCys: false }).every((r) => r.mutation === "A"),
  );
  const csv = toCSV(result);
  assert.ok(csv.includes('"A","3","A"'));
  assert.equal(csv.trim().split("\n").length, 201);
  assert.ok(toFASTA(result).startsWith(">ThermoMPNN_chain_A\n"));
  assert.equal(centrality(s).length, s.length);
  assert.equal(centrality(s)[3], -1);
  assert.throws(() => centrality(s, { radius: -1 }));
});

test("Exact PyTorch topk tie order, from 1 to 2048 positions", async () => {
  for (const c of await readJSON("fixtures/topk_cases.json"))
    assert.deepEqual(
      [...topk(Float32Array.from(c.values), Math.min(48, c.values.length))],
      c.indices,
      c.name,
    );
});

test("Per-chain centrality matches the exact upstream analysis helper, including missing atoms", async () => {
  const expected = await readJSON("fixtures/centrality.json");
  for (const c of await readJSON("fixtures/cases.json")) {
    const s = parseStructure(await text(c.input), {
      chains: c.inputChains ?? c.chains,
    });
    assert.deepEqual([...centrality(s)], expected[c.same_as ?? c.id], c.id);
  }
});
