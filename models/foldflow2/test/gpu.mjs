import { mkdir, writeFile } from "node:fs/promises";
import { runBrowser } from "./browser.mjs";
for (const d of ["tmp", "browser-cache", "browser-config"])
  await mkdir(new URL("../.cache/" + d, import.meta.url), { recursive: true });
const result = await runBrowser("/test/gpu-suite.js", { timeoutMs: 3600000 });
await writeFile(
  new URL("../fixtures/gpu_metrics.json", import.meta.url),
  JSON.stringify(result, null, 2),
);
if (!result.ok) throw Error(result.error);
console.log(JSON.stringify(result, null, 2));
