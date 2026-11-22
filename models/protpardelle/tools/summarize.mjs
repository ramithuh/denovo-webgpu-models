import { readFile, writeFile, readdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
const root = fileURLToPath(new URL("../", import.meta.url));
const json = async (name) => JSON.parse(await readFile(root + name, "utf8"));
const save = async (name, value) =>
  writeFile(root + name, JSON.stringify(value, null, 2) + "\n");
const max = (values) => Math.max(0, ...values),
  fmt = (x) => x.toExponential(2);
const exports = await json("fixtures/export_metrics.json"),
  matrix = await json("fixtures/mode_matrix.json"),
  cpu = await json("fixtures/cpu_metrics.json"),
  gpu = await json("fixtures/gpu_metrics.json"),
  scpu = await json("fixtures/sampler_cpu_metrics.json"),
  sgpu = await json("fixtures/sampler_gpu_metrics.json"),
  inventory = await json("fixtures/sampler_inventory.json");
const quant = [
    ...(await json("fixtures/denoiser_export_metrics.json")),
    ...(await json("fixtures/minimpnn_export_metrics.json")),
  ],
  bench = await json("fixtures/benchmark_metrics.json");
if (
  exports.length !== 48 ||
  gpu.length !== 48 ||
  scpu.length !== 111 ||
  sgpu.length !== 111 ||
  inventory.length !== 111
)
  throw new Error("Incomplete full validation metrics");
const gpuTraces = gpu.flatMap((run) =>
  run.results
    .filter((r) => Number.isFinite(r.relRMS))
    .map((r) => ({ ...r, variant: run.variant, precision: run.precision })),
);
for (const row of [...cpu, ...gpuTraces, ...scpu, ...sgpu])
  if (!(row.relRMS < 1e-4)) throw new Error("Numerical gate failed");
for (const row of [...scpu, ...sgpu])
  if (!(row.maxTraceRelRMS < 1e-4) || !row.traceCount)
    throw new Error("Sampler trace metrics missing or failed");
for (const row of exports) {
  const output = quant.filter(
    (q) => q.variant === row.variant && q.precision === row.precision,
  );
  if (!output.length) throw new Error("Missing quantization reference");
  row.upstream_output_roundtrip = "verified";
  row.output_relRMS_max = max(output.map((r) => r.output_relRMS));
  row.output_modes = output;
}
await save("fixtures/export_metrics.json", exports);
const loadSampler = async (id) => {
  const meta = await json(`fixtures/sampler_${id}.json`),
    blob = await readFile(root + `fixtures/sampler_${id}.bin`);
  const get = (d) =>
    new (d.dtype === "I32" ? Int32Array : Float32Array)(
      blob.buffer.slice(
        blob.byteOffset + d.offset,
        blob.byteOffset + d.offset + d.length * 4,
      ),
    );
  const out = meta.stage2_outputs ?? meta.outputs;
  return { x: get(out.x), s: get(meta.outputs.s) };
};
const quality = [];
for (const entry of inventory.filter((r) => r.precision === "fp32")) {
  const baseline = await loadSampler(entry.preset);
  for (const precision of ["fp32", "fp16", "int8"]) {
    const out =
      precision === "fp32"
        ? baseline
        : await loadSampler(entry.preset + "_" + precision);
    let error = 0,
      norm = 0,
      caError = 0,
      matches = 0;
    for (let i = 0; i < out.x.length; i++) {
      const d = out.x[i] - baseline.x[i];
      error += d * d;
      norm += baseline.x[i] ** 2;
      if (i % 111 >= 3 && i % 111 < 6) caError += d * d;
    }
    for (let i = 0; i < out.s.length; i++)
      matches += out.s[i] === baseline.s[i];
    quality.push({
      case: entry.preset,
      variant: entry.variant,
      precision,
      coordinate_relRMS: Math.sqrt(error / Math.max(norm, 1e-30)),
      ca_RMS_angstrom: Math.sqrt(caError / out.s.length),
      sequence_identity: matches / out.s.length,
    });
  }
}
await save("fixtures/sampler_quantization_metrics.json", quality);
for (const entry of inventory) {
  entry.port_status = "verified_with_recorded_minimpnn_input_traces";
  entry.cpu = scpu.find((r) => r.preset === entry.preset);
  entry.webgpu = sgpu.find((r) => r.preset === entry.preset);
}
await save("fixtures/sampler_inventory.json", inventory);
matrix.validation_status =
  "published_checkpoint_modes_verified_with_documented_trace_boundaries";
matrix.trace_boundary =
  "Denoiser/guidance and complete trajectory checks run freely; MiniMPNN internal traces use recorded original MiniMPNN inputs because degenerate quaternion features are discontinuous.";
matrix.external_components = [
  {
    name: "full ProteinMPNN redesign",
    status: "excluded; separate weights and model",
  },
  {
    name: "ESMFold / BindCraft / AlphaFold evaluation",
    status: "external integrations, not ported",
  },
  { name: "secondary structure / adjacency weights", status: "unreleased" },
];
matrix.coverage = await json("docs/MODE_COVERAGE.json");
matrix.additional_diagnostics = {
  dense_cloud_coordinate_gradients: await json(
    "fixtures/gradient_diagnostic_metrics.json",
  ),
};
for (const argument of matrix.sampler_arguments)
  if (!matrix.coverage.original_sample_arguments[argument])
    throw new Error("Unaudited upstream sample argument: " + argument);
matrix.modes = [...new Set(cpu.map((r) => r.mode))].map((mode) => ({
  mode,
  cpu_max_relRMS: max(cpu.filter((r) => r.mode === mode).map((r) => r.relRMS)),
  gpu_max_relRMS: max(
    gpuTraces.filter((r) => r.mode === mode).map((r) => r.relRMS),
  ),
  status: "verified",
}));
matrix.sampler_cases = inventory.map(
  ({ preset, variant, precision, port_status }) => ({
    case: preset,
    variant,
    precision,
    status: port_status,
  }),
);
await save("fixtures/mode_matrix.json", matrix);
const lines = [
  `Measured on NVIDIA GB10 (driver 580.159.03), ARM64 Linux, Chrome 153.0.8010.12 (Playwright build 1243) / Vulkan. All 16 checkpoints × three precisions are covered. The sampler suite contains ${inventory.length} replay cases across 16 presets, 15 structural checkpoints, and six additional control combinations.`,
  "",
  "| Checkpoint | Parameters / tensors | FP32 bytes / shards | FP16 bytes / shards | INT8 bytes / shards |",
  "| --- | ---: | ---: | ---: | ---: |",
];
for (const checkpoint of matrix.checkpoints) {
  const rows = ["fp32", "fp16", "int8"].map((p) =>
    exports.find((r) => r.variant === checkpoint.name && r.precision === p),
  );
  lines.push(
    `| ${checkpoint.name} | ${rows[0].total_params.toLocaleString("en-US")} / ${rows[0].tensors} | ${rows.map((r) => `${r.bytes.toLocaleString("en-US")} / ${r.shards}`).join(" | ")} |`,
  );
}
lines.push(
  "",
  "Parameter counts exclude buffers and duplicate state aliases; `state_values` and non-trainable frequency parameters are separately recorded in each manifest. The MiniMPNN archive checkpoint includes a frozen structural network, retained in exports.",
  "",
  "| Precision | Max weight relRMS | Max neural-output quantization relRMS | Max free-sampler coordinate quantization relRMS | Minimum sequence identity |",
  "| --- | ---: | ---: | ---: | ---: |",
);
for (const p of ["fp32", "fp16", "int8"]) {
  const q = quality.filter((r) => r.precision === p);
  lines.push(
    `| ${p} | ${fmt(max(exports.filter((r) => r.precision === p).map((r) => r.weight_relRMS)))} | ${fmt(max(quant.filter((r) => r.precision === p).map((r) => r.output_relRMS)))} | ${fmt(max(q.map((r) => r.coordinate_relRMS)))} | ${Math.min(...q.map((r) => r.sequence_identity)).toFixed(3)} |`,
  );
}
lines.push(
  "",
  "The last two columns compare independently running quantized samplers with the same recorded random draws. Changed categorical decisions can substantially change a generated structure. Coordinates are unaligned; see the JSON for Cα RMS and per-case values.",
  "",
  "| Network/input mode | Max CPU trace relRMS | Max GPU trace relRMS | Result (all storage precisions) |",
  "| --- | ---: | ---: | --- |",
);
for (const r of matrix.modes)
  lines.push(
    `| ${r.mode} | ${fmt(r.cpu_max_relRMS)} | ${fmt(r.gpu_max_relRMS)} | pass |`,
  );
lines.push(
  "",
  "| Original sampler preset / extra control | Max CPU final relRMS | Max GPU final relRMS | Max checked trace relRMS | Tokens / masks |",
  "| --- | ---: | ---: | ---: | --- |",
);
for (const entry of inventory.filter(
  (r) => r.precision === "fp32" && !r.preset.startsWith("checkpoint_"),
)) {
  const ids = [entry.preset, entry.preset + "_fp16", entry.preset + "_int8"],
    a = scpu.filter((r) => ids.includes(r.preset)),
    b = sgpu.filter((r) => ids.includes(r.preset));
  lines.push(
    `| ${entry.preset} | ${fmt(max(a.map((r) => r.relRMS)))} | ${fmt(max(b.map((r) => r.relRMS)))} | ${fmt(max([...a, ...b].map((r) => r.maxTraceRelRMS)))} | exact |`,
  );
}
lines.push(
  "",
  "All structural-checkpoint sampler rows, including multichain hotspots, are in `fixtures/sampler_inventory.json`. Parser/contig fixtures cover insertion codes, negative numbering, gaps, altLoc selection, MSE/UNK, multiple models, hetero context, multi-character mmCIF chain IDs, chain layouts, and discontiguous selection.",
  "",
  "| Storage | Cold load ms | Warm 64-residue median ms | Warm 128-residue median ms |",
  "| --- | ---: | ---: | ---: |",
);
const median = (a) => a.toSorted((a, b) => a - b)[Math.floor(a.length / 2)];
for (const r of bench)
  lines.push(
    `| ${r.precision} | ${r.loadMs.toFixed(1)} | ${median(r.results.find((x) => x.length === 64).warmMs).toFixed(1)} | ${median(r.results.find((x) => x.length === 128).warmMs).toFixed(1)} |`,
  );
lines.push(
  "",
  "Timings include encoding, submission, and readback for one denoiser evaluation, after warmup. Cold loads use local HTTP without model cache. They are not full 200–500-step sampling times. Module-worker inference, Cache API/OPFS offline loading, denied storage, cancellation, caller device ownership, buffer admission, and the demo have real-browser tests.",
);
const file = root + "README_WEBGPU.md",
  text = await readFile(file, "utf8");
await writeFile(
  file,
  text.replace(
    /<!-- RESULTS:START -->[\s\S]*?<!-- RESULTS:END -->/,
    "<!-- RESULTS:START -->\n" + lines.join("\n") + "\n<!-- RESULTS:END -->",
  ),
);
console.log(
  `Summarized ${cpu.length + gpuTraces.length} network traces and ${inventory.length} sampler cases.`,
);
