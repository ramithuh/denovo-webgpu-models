import { execFileSync } from "node:child_process";
import { writeFile } from "node:fs/promises";
import { runBrowser } from "./browser.mjs";
const result = await runBrowser("/test/gpu-suite.js", { timeoutMs: 1800000 });
if (!result.ok) throw Error(result.error);
result.hostGPU = execFileSync(
  "nvidia-smi",
  ["--query-gpu=name,driver_version,uuid", "--format=csv,noheader"],
  { encoding: "utf8" },
).trim();
await writeFile(
  new URL("../fixtures/gpu_metrics.json", import.meta.url),
  JSON.stringify(result, null, 2) + "\n",
);
console.log(JSON.stringify(result, null, 2));
