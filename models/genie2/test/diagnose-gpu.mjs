import { runBrowser } from "./browser.mjs";
import { writeFile } from "node:fs/promises";
const r = await runBrowser("/test/diagnose-gpu.js", { timeoutMs: 900000 });
console.log(r);
await writeFile(
  new URL("../.cache/gpu-diagnostic.json", import.meta.url),
  JSON.stringify(r, null, 2),
);
if (!r.ok) process.exitCode = 1;
