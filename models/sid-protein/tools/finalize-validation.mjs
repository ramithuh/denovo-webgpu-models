import { readFile, writeFile, readdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
const root = new URL("../", import.meta.url),
  read = async (path) => readFile(new URL(path, root)),
  json = async (path) => JSON.parse(await read(path));
const gpu = await json("fixtures/gpu_metrics.json"),
  cpu = await json("fixtures/cpu_metrics.json");
if (!gpu.ok || !cpu.ok || gpu.skipped !== 0 || cpu.skipped !== 0)
  throw new Error("Verification incomplete");
for (const row of [...gpu.rows, ...cpu.rows])
  if (
    row.forward.relRms >= 1e-4 ||
    row.sampling.relRms >= 1e-4 ||
    row.trajectory.relRms >= 1e-4
  )
    throw new Error("Parity threshold failed");
const hash = createHash("sha256");
const sourceFiles = [];
for (const dir of ["shared", "cpu", "webgpu"])
  for (const name of (await readdir(new URL(dir + "/", root))).sort()) {
    if (!name.endsWith(".js")) continue;
    const path = dir + "/" + name;
    sourceFiles.push(path);
    hash.update(path + "\n").update(await read(path));
  }
const runtimeSourceSha256 = hash.digest("hex"),
  logs = {};
for (const name of ["npm-test.log", "gpu-release.log"])
  logs[name] = createHash("sha256")
    .update(await read(".cache/" + name))
    .digest("hex");
const validation = {
  ok: true,
  skipped: 0,
  timestamp: new Date().toISOString(),
  upstreamCommit: execFileSync(
    "git",
    ["-C", new URL(".cache/upstream", root).pathname, "rev-parse", "HEAD"],
    { encoding: "utf8" },
  ).trim(),
  runtimeSourceSha256,
  sourceFiles,
  commands: ["npm run test:gpu", "npm test"],
  logs,
  gpuCases: gpu.rows.length,
  cpuCases: cpu.rows.length,
  gpuHost: gpu.hostGPU,
  contracts: gpu.contracts,
};
await writeFile(
  new URL("fixtures/validation.json", root),
  JSON.stringify(validation, null, 2) + "\n",
);
const fmt = (x) => x.toExponential(3),
  ms = (x) => x.toFixed(1),
  lines = [];
lines.push(
  `Verified on **${gpu.hostGPU.split(",").slice(0, 2).join(",")}**, ${validation.timestamp}. Both \`npm run test:gpu\` and \`npm test\` passed, with **zero skipped tests**. CPU validation covers all six forward/trajectory cases; GPU validation covers all 18 precision/case combinations plus the original sensitivity regression.`,
);
lines.push(
  "\n| Precision | Worst GPU forward relRMS | Worst GPU final relRMS | Worst GPU trajectory relRMS | Packed GPU weight bytes |\n| --- | ---: | ---: | ---: | ---: |",
);
for (const precision of ["fp32", "fp16", "int8"]) {
  const rows = gpu.rows.filter((r) => r.precision === precision);
  lines.push(
    `| ${precision.toUpperCase()} | ${fmt(Math.max(...rows.map((r) => r.forward.relRms)))} | ${fmt(Math.max(...rows.map((r) => r.sampling.relRms)))} | ${fmt(Math.max(...rows.map((r) => r.trajectory.relRms)))} | ${gpu.storage[precision].weightBytes.toLocaleString("en-US")} |`,
  );
}
lines.push(
  `\nWorst CPU final-coordinate relRMS: **${fmt(Math.max(...cpu.rows.map((r) => r.sampling.relRms)))}**. Packed weight ratios are FP16 **${gpu.ratios.fp16.toFixed(6)}×** and INT8 **${gpu.ratios.int8.toFixed(6)}×** relative to FP32.`,
);
lines.push(
  "\n| Precision / length | First forward (ms) | Warm forward (ms) | Full 16-step sampling (ms) | Quantization drift vs FP32 relRMS |\n| --- | ---: | ---: | ---: | ---: |",
);
for (const row of gpu.rows.filter((r) => r.name.endsWith("T16")))
  lines.push(
    `| ${row.precision.toUpperCase()} / ${row.name.split("_")[0]} | ${ms(row.forwardColdMs)} | ${ms(row.forwardWarmMs)} | ${ms(row.samplingMs)} | ${fmt(row.quantizationDrift.relRms)} |`,
  );
lines.push(
  "\n“First forward” includes shader compilation only for kernels not already compiled by that engine; the first L=32 row is cold, and L=64 reuses the same dynamic pipelines. Timings include queue execution and readback, exclude weight loading, and are measurements on this shared verification host rather than throughput guarantees. Quantization drift describes changed coordinates, not designability. FP32 is the appropriate choice when preserving the original checkpoint trajectory matters.",
);
lines.push(
  `\nThe original L=64, 8-step, seed-1298 regression now passes at **${fmt(gpu.contracts.sensitivityRegression.sampling.relRms)}** final relRMS. The fixes preserve biased-linear accumulation order, PyTorch's Welford and coordinate-sum reductions, SLEEF transcendental arithmetic (including four-lane positional argument reduction), and separate interpolation rounding. No tolerance was relaxed, no generated coordinates are replayed, and no rigid alignment is applied.`,
);
lines.push(
  "\nHardware tests also pass for Web Worker forward/full sampling and cancellation; Cache API and OPFS cold/warm reads; simulated Safari Private Mode denial; corrupted-shard rejection; multipart INT8 scale colocation; external GPUDevice ownership; buffer-limit rejection; packed half decoding without shader-f16; and normal/error/abort cleanup. Transient buffers are zero after calls and engine buffer bytes are zero after destruction.",
);
lines.push(
  `\nComplete evidence: [GPU metrics](fixtures/gpu_metrics.json), [CPU metrics](fixtures/cpu_metrics.json), [export metrics](fixtures/export_metrics.json), and [validation provenance](fixtures/validation.json). Runtime source SHA-256: \`${runtimeSourceSha256}\`.`,
);
let doc = (await read("README_WEBGPU.md")).toString();
doc = doc.replace(
  /<!-- VERIFIED_RESULTS -->[\s\S]*?<!-- END_VERIFIED_RESULTS -->/,
  "<!-- VERIFIED_RESULTS -->\n" +
    lines.join("\n") +
    "\n<!-- END_VERIFIED_RESULTS -->",
);
await writeFile(new URL("README_WEBGPU.md", root), doc);
console.log(
  JSON.stringify(
    {
      runtimeSourceSha256,
      gpuWorst: Math.max(...gpu.rows.map((r) => r.sampling.relRms)),
      cpuWorst: Math.max(...cpu.rows.map((r) => r.sampling.relRms)),
    },
    null,
    2,
  ),
);
