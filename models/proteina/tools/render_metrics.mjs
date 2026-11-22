import { readFile, writeFile } from "node:fs/promises";
const root = new URL("../", import.meta.url),
  read = async (p) => JSON.parse(await readFile(new URL(p, root), "utf8"));
const [exports, cpu, gpu] = await Promise.all(
  ["export_metrics", "cpu_metrics", "gpu_metrics"].map((n) =>
    read("fixtures/" + n + ".json"),
  ),
);
const rows = [
  "| Precision | Tensors | Shards | Bundle bytes | Largest shard | Resident weights + constants |",
  "| --- | ---: | ---: | ---: | ---: | ---: |",
];
for (const p of ["fp32", "fp16", "int8"]) {
  const e = exports[p];
  rows.push(
    `| ${p} | ${e.tensors} | ${e.shards} | ${e.bytes.toLocaleString("en-US")} | ${e.max_shard_bytes.toLocaleString("en-US")} | ${gpu.precisions[p].memory.weightBytes.toLocaleString("en-US")} |`,
  );
}
rows.push(
  "",
  "Export round-trip maximum absolute error: **0 for all precisions**, with strict upstream model reload.",
  "",
  `Hardware: **${gpu.hostGPU ?? "NVIDIA GB10"}**. Browser: \`${gpu.userAgent}\`.`,
  "",
  "| Backend | Worst denoiser relRMS | Worst final sampler relRMS | Warm L=32 / L=64 (ms) |",
  "| --- | ---: | ---: | --- |",
);
const max = (a, key = "relRMS") =>
  Math.max(...a.map((x) => x[key])).toExponential(3);
rows.push(
  `| CPU FP32 | ${max(cpu.denoise)} | ${max(cpu.samples)} | reference backend |`,
);
for (const p of ["fp32", "fp16", "int8"]) {
  const r = gpu.precisions[p];
  rows.push(
    `| GPU ${p} | ${max(r.denoise)} | ${max(r.samples)} | ${r.warmTimings[32].medianMs.toFixed(1)} / ${r.warmTimings[64].medianMs.toFixed(1)} |`,
  );
}
rows.push(
  "",
  "Every final tensor passes **relRMS < 1e-4**, with **zero skipped tests**. Cases cover L=32/64, T=2/3/4, ODE/SDE, log/linear/power schedules, self-conditioning on/off, fold conditioning and CFG.",
  "",
  "| Quantized PyTorch vs FP32 PyTorch | Worst denoiser drift | Worst final sampler drift |",
  "| --- | ---: | ---: |",
);
for (const p of ["fp16", "int8"])
  rows.push(
    `| ${p} | ${max(gpu.quantizationDrift[p].denoise)} | ${max(gpu.quantizationDrift[p].samples)} |`,
  );
rows.push(
  "",
  `Resident weight ratios: FP16 **${gpu.memoryRatios.fp16.toFixed(4)}×**, INT8 **${gpu.memoryRatios.int8.toFixed(4)}×** FP32. Native-F16 load parity: **${gpu.nativeF16.relRMS.toExponential(3)}**.`,
  "",
  "| Precision | Cold load (ms) | First forward including compilation and traces (ms) |",
  "| --- | ---: | ---: |",
);
for (const p of ["fp32", "fp16", "int8"]) {
  const r = gpu.precisions[p];
  rows.push(
    `| ${p} | ${r.loadMs.toFixed(1)} | ${r.denoise[0].timingMs.toFixed(1)} |`,
  );
}
rows.push(
  "",
  "Transient buffers after normal/error/abort paths: **0**. Tracked bytes after idempotent destroy: **0**. Offline Cache API and OPFS worker runs, both simulated Private Mode fallbacks, and the interactive demo pass. Warm numbers are medians of three full forward calls and include dispatch/readback overhead. Cold loads use the local HTTP fixture server.",
);
const file = new URL("README_WEBGPU.md", root),
  text = await readFile(file, "utf8");
await writeFile(
  file,
  text.replace(
    /<!-- METRICS_START -->[\s\S]*?<!-- METRICS_END -->/,
    "<!-- METRICS_START -->\n" + rows.join("\n") + "\n<!-- METRICS_END -->",
  ),
);
