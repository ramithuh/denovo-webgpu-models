import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  buildFeatures,
  sampleScaffold,
  parsePDB,
  writePDB,
  parseMotifSpecification,
  diffusionSchedule,
  seededRandom,
  loadWeights,
} from "../shared/index.js";
import { read, root, fixture } from "./fixtures.mjs";
test("Official bundles: exact quantized round-trip, bounded authenticated shards", async () => {
  const metrics = JSON.parse(
    await readFile(new URL("fixtures/export_metrics.json", root), "utf8"),
  );
  for (const p of ["fp32", "fp16", "int8"]) {
    assert.equal(metrics[p].num_tensors, 396);
    assert.equal(metrics[p].total_params, 15732080);
    assert.equal(metrics[p].roundtrip.maxAbs, 0);
    for (const s of metrics[p].shards) assert(s.byteLength <= 25000000);
    const b = await loadWeights(`weights/${p}`, { read });
    assert.equal(Object.keys(b.weights).length, 396);
  }
  await assert.rejects(
    loadWeights("weights/fp32", {
      read: async (url) => {
        const b = await read(url);
        if (url.endsWith(".safetensors"))
          new Uint8Array(b)[b.byteLength - 1] ^= 1;
        return b;
      },
    }),
    /checksum/,
  );
});
test("Single and multi-motif group masking, sampled scaffold constraints and C-alpha PDB", () => {
  const motifs = [
      { start: 2, group: 1, sequence: "AR", coordinates: [0, 0, 0, 3.8, 0, 0] },
      { start: 8, group: 2, sequence: "ND", coordinates: [0, 2, 0, 3.8, 2, 0] },
    ],
    f = buildFeatures({ length: 14, motifs });
  assert.equal(f.fixed_structure_mask[2 * 14 + 3], 1);
  assert.equal(f.fixed_structure_mask[2 * 14 + 8], 0);
  const xyz = Float32Array.from({ length: 42 }, (_, i) => i * 0.2),
    pdb = writePDB({ coordinates: xyz, center: false }),
    r = parsePDB(pdb);
  assert.equal(r.sequence.length, 14);
  assert(Math.abs(r.coordinates[41] - xyz[41]) < 0.001);
  const segments = [
    { min: 2, max: 8 },
    { motif: motifs[0] },
    { min: 3, max: 7 },
  ];
  for (let i = 0; i < 20; i++) {
    const s = sampleScaffold(
      { segments, minLength: 11, maxLength: 14 },
      seededRandom(i),
    );
    assert(s.length >= 11 && s.length <= 14);
    assert.equal(
      buildFeatures(s).fixed_sequence_mask.reduce((a, b) => a + b),
      2,
    );
  }
  assert.throws(
    () => sampleScaffold({ segments, minLength: 100, maxLength: 101 }),
    /no valid/,
  );
  assert.throws(
    () => buildFeatures({ length: 10, motifs: [motifs[0], motifs[0]] }),
    /Overlapping/,
  );
  const spec = parseMotifSpecification(
    "REMARK 999 INPUT      3  5\nREMARK 999 INPUT  A   1   2 A\nREMARK 999 INPUT      4  8\n" +
      pdb,
  );
  assert.equal(spec.segments.length, 3);
});
test("Dynamic cosine schedules match PyTorch float32 coefficients at distinct step counts", async () => {
  for (const [index, steps] of [
    [0, 2],
    [3, 3],
  ]) {
    const { get } = await fixture("fp32", index),
      actual = diffusionSchedule(steps).beta,
      expected = get("betas");
    assert.equal(actual.length, expected.length);
    for (let i = 0; i < actual.length; i++)
      assert(Math.abs(actual[i] - expected[i]) < 2e-7);
  }
  assert.throws(() => diffusionSchedule(0));
});
