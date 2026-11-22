import { mkdir, writeFile } from "node:fs/promises";
import { runBrowser } from "./browser.mjs";
import { execFileSync } from "node:child_process";
for (const d of ["tmp", "browser-cache", "browser-config"])
  await mkdir(new URL("../.cache/" + d, import.meta.url), { recursive: true });
await mkdir(new URL("../.t", import.meta.url), { recursive: true });
const result = await runBrowser("/test/gpu-suite.js", { timeoutMs: 1800000 });
if (!result.ok) throw new Error(result.error);
result.hostGPU = execFileSync(
  "nvidia-smi",
  ["--query-gpu=name,driver_version,uuid", "--format=csv,noheader"],
  { encoding: "utf8" },
).trim();
if (!result.hostGPU.includes("NVIDIA GB10"))
  throw new Error("Expected NVIDIA GB10 on this verification host");
await writeFile(
  new URL("../fixtures/gpu_metrics.json", import.meta.url),
  JSON.stringify(result, null, 2) + "\n",
);
console.log(JSON.stringify(result, null, 2));
console.log("Hardware WebGPU: complete forward + sampling passed; zero skips.");
