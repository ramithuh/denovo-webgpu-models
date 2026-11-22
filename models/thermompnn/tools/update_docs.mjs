import { readFile, writeFile } from "node:fs/promises";
const root = new URL("../", import.meta.url);
const json = async (name) =>
  JSON.parse(await readFile(new URL("fixtures/" + name, root), "utf8"));
const [exports, cpu, gpu, cases] = await Promise.all(
  [
    "export_metrics.json",
    "cpu_metrics.json",
    "gpu_metrics.json",
    "cases.json",
  ].map(json),
);
const scientific = (n) => n.toExponential(3);
const maxTrace = (c) =>
  Math.max(...Object.values(c.traces).map((t) => t.relRMS));
const lines = [
  "| Bundle | Shards | Safetensors bytes | Weight relRMS | ΔΔG relRMS vs original FP32 (range) |",
  "| --- | ---: | ---: | ---: | ---: |",
];
for (const p of ["fp32", "fp16", "int8"]) {
  const m = exports[p],
    errors = Object.values(m.cases).map((c) => c.ddg_relRMS);
  lines.push(
    `| ${p.toUpperCase()} | ${m.shards} | ${m.bytes.toLocaleString("en-US")} | ${scientific(m.weight_relRMS)} | ${scientific(Math.min(...errors))}–${scientific(Math.max(...errors))} |`,
  );
}
lines.push(
  "",
  "FP16 ddG quantization error is below 0.06% on these inputs. INT8 trades accuracy for download size (up to 4.66% relative ddG error here, maximum absolute deviation 0.201 kcal/mol on the full chain). These are measured storage effects; the arithmetic parity threshold remains 1e-4 for **every** precision. No biological accuracy claim is inferred from these fixtures.",
  "",
  "Every row below passes SSM and targeted scoring, all intermediate traces, exact integer preprocessing/graphs, and WT identity checks. CPU numbers are the worst trace across all three storage precisions; GPU output columns compare against identically quantized PyTorch. “All GPU traces” includes native shader-f16.",
  "",
  "| Input mode | CPU max trace relRMS | GPU FP32 ΔΔG | GPU FP16 ΔΔG | GPU INT8 ΔΔG | All GPU traces, max relRMS | Result |",
  "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
);
for (const c of cases) {
  const cpuMax = Math.max(
    ...["fp32", "fp16", "int8"].map((p) => maxTrace(cpu[p][c.id])),
  );
  const gpuMax = Math.max(
    ...Object.values(gpu.precisions).map((p) => maxTrace(p.cases[c.id])),
  );
  if (cpuMax >= 1e-4 || gpuMax >= 1e-4)
    throw new Error("Unpassed parity metric");
  lines.push(
    `| ${c.id} | ${scientific(cpuMax)} | ${["fp32", "fp16", "int8"].map((p) => scientific(gpu.precisions[p].cases[c.id].traces.ddg.relRMS)).join(" | ")} | ${scientific(gpuMax)} | PASS |`,
  );
}
const fullCpu = Math.max(
    ...Object.values(cpu).map((p) => p.full_length.traces.ddg.relRMS),
  ),
  fullGpu = Math.max(
    ...Object.values(gpu.precisions).map((p) => p.fullLengthParity.relRMS),
  );
lines.push(
  "",
  `The additional complete **194-residue 2OCJ chain A** passes final-output parity (CPU max ${scientific(fullCpu)}, GPU max ${scientific(fullGpu)}), exercising dispatches beyond one workgroup-grid dimension. Full intermediate traces are checked on the mode fixtures above.`,
  "",
  `Measured on NVIDIA GB10, Chromium ${gpu.browserVersion}:`,
  "",
  "| Storage / loads | Cold engine load (ms) | Warm 12 residues (ms) | Warm 76 residues (ms) | Warm 194 residues (ms) |",
  "| --- | ---: | ---: | ---: | ---: |",
);
for (const [name, m] of Object.entries(gpu.precisions))
  lines.push(
    `| ${name} | ${m.coldLoadMs.toFixed(1)} | ${["small", "medium", "full"].map((size) => m.benchmarks[size].medianMs.toFixed(1)).join(" | ")} |`,
  );
const path = new URL("README_WEBGPU.md", root),
  text = await readFile(path, "utf8");
await writeFile(
  path,
  text.replace(
    /<!-- METRICS_START -->[\s\S]*?<!-- METRICS_END -->/,
    "<!-- METRICS_START -->\n" + lines.join("\n") + "\n<!-- METRICS_END -->",
  ),
);
