import { execFileSync } from "node:child_process";
import { buildLoaderFixture } from "./loader-fixture.mjs";
import { mkdir, writeFile } from "node:fs/promises";
import { runBrowser } from "./browser.mjs";
for (const d of ["tmp", "browser-cache", "browser-config"])
  await mkdir(new URL("../.cache/" + d, import.meta.url), { recursive: true });
await buildLoaderFixture();
const result = await runBrowser("/test/gpu-suite.js", { timeoutMs: 3600000 });
if (!result.ok) throw new Error(result.error);
result.hostGPU = execFileSync(
  "nvidia-smi",
  ["--query-gpu=name,driver_version,uuid", "--format=csv,noheader"],
  { encoding: "utf8" },
).trim();
result.timestamp = new Date().toISOString();
await writeFile(
  new URL("../fixtures/gpu_metrics.json", import.meta.url),
  JSON.stringify(result, null, 2) + "\n",
);
console.log(JSON.stringify(result, null, 2));
