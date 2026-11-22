import assert from "node:assert/strict";
import test from "node:test";
import { CpuOps, LigandModel } from "../cpu/index.js";
import { parseStructure, featurize } from "../shared/ligandmpnn/index.js";
import { meta, readOracle, loadWeights, structure, stats } from "./helpers.mjs";
function compare(name, got, ref, tol = 1e-4) {
  const s = stats(got, ref);
  assert.ok(s.relRMS < tol, `${name}: ${JSON.stringify(s)}`);
  return s;
}
const parsed = parseStructure(structure),
  input = featurize(parsed, meta.options);
test("PDB parsing, context selection and protein graph match upstream", () => {
  for (const key of [
    "X",
    "Y",
    "Y_t",
    "Y_m",
    "S",
    "mask",
    "R_idx",
    "chain_labels",
    "xyz_37",
    "xyz_37_m",
  ])
    compare("parsed." + key, parsed[key], readOracle("parsed." + key), 1e-7);
  for (const key of [
    "X",
    "Y",
    "Y_t",
    "Y_m",
    "S",
    "mask",
    "R_idx",
    "chain_labels",
    "chain_mask",
    "mask_XY",
    "bias",
  ])
    compare("input." + key, input[key], readOracle("input." + key), 1e-7);
  compare("virtual CB", input.CB, readOracle("parsed.CB"), 1e-7);
  assert.deepEqual(
    Array.from(input.E_idx),
    Array.from(readOracle("score.features.0.E_idx")),
  );
});
test("FP32 features, encoders and teacher-forced decoder parity", async () => {
  const model = new LigandModel(new CpuOps(), loadWeights());
  const out = await model.score(input, {
    randn: readOracle("input.randn"),
    trace: true,
  });
  let worst = { name: "", relRMS: 0 };
  for (const [key, value] of Object.entries(out.trace)) {
    const s = compare(key, value, readOracle("score." + key));
    if (s.relRMS > worst.relRMS) worst = { name: key, ...s };
  }
  compare("logits", out.logits, readOracle("score.output.logits"));
  compare("log_probs", out.log_probs, readOracle("score.output.log_probs"));
  console.log("CPU score worst:", worst);
});
for (const which of ["sample", "tied", "sidechain", "no_context"])
  test("FP32 autoregressive replay: " + which, async () => {
    const opts = { ...meta.options };
    if (which === "tied")
      opts.tiedPositions = [
        [2, 3],
        [4, 5],
      ];
    if (which === "sidechain") opts.useSideChainContext = true;
    if (which === "no_context") opts.useAtomContext = false;
    const model = new LigandModel(new CpuOps(), loadWeights());
    const start = performance.now();
    const out = await model.sample(featurize(parsed, opts), {
      temperature: meta.options.temperature,
      randn: readOracle("input.randn"),
      uniforms: readOracle(which + ".uniforms"),
      trace: true,
    });
    let worst = { name: "", relRMS: 0 };
    for (const [key, value] of Object.entries(out.trace)) {
      if (key.startsWith("encoded.")) continue;
      const s = compare(key, value, readOracle(which + "." + key));
      if (s.relRMS > worst.relRMS) worst = { name: key, ...s };
    }
    for (const key of ["sampling_probs", "log_probs"])
      compare(key, out[key], readOracle(which + ".output." + key));
    assert.deepEqual(
      Array.from(out.S),
      Array.from(readOracle(which + ".output.S")),
    );
    assert.deepEqual(
      Array.from(out.decoding_order),
      Array.from(readOracle(which + ".output.decoding_order")),
    );
    compare(
      "global_score",
      [out.global_score],
      readOracle(which + ".global_score"),
    );
    compare(
      "ligand_score",
      [out.ligand_score],
      readOracle(which + ".ligand_score"),
    );
    console.log(
      which,
      JSON.stringify({
        worst,
        sequence: out.sequence,
        ms: performance.now() - start,
      }),
    );
  });
test("Independent small-molecule + zinc holo PyTorch parity", async () => {
  const fs = await import("node:fs");
  const m = JSON.parse(
      fs.readFileSync(
        new URL("../fixtures/oracle_small_molecule.json", import.meta.url),
      ),
    ),
    b = fs.readFileSync(
      new URL("../fixtures/oracle_small_molecule.bin", import.meta.url),
    );
  const read = (name) => {
    const t = m.tensors[name],
      raw = b.buffer.slice(
        b.byteOffset + t.byteOffset,
        b.byteOffset + t.byteOffset + t.byteLength,
      );
    return t.dtype === "I32" ? new Int32Array(raw) : new Float32Array(raw);
  };
  const p = parseStructure(
    fs.readFileSync(
      new URL("../fixtures/" + m.structure, import.meta.url),
      "utf8",
    ),
  );
  for (const k of ["X", "Y", "Y_t", "Y_m", "S", "mask", "xyz_37", "xyz_37_m"])
    compare("small parsed." + k, p[k], read("parsed." + k), 1e-7);
  const f = featurize(p, m.options),
    model = new LigandModel(new CpuOps(), loadWeights());
  const score = await model.score(f, { randn: read("input.randn") });
  compare("small logits", score.logits, read("score.logits"));
  const out = await model.sample(f, {
    randn: read("input.randn"),
    uniforms: read("uniforms"),
    temperature: 0.3,
  });
  assert.deepEqual(Array.from(out.S), Array.from(read("sample.S")));
  compare("small log_probs", out.log_probs, read("sample.log_probs"));
  console.log(
    "small molecule",
    JSON.stringify({
      logits: stats(score.logits, read("score.logits")),
      sequence: out.sequence,
    }),
  );
});
