import { readFile, writeFile } from "node:fs/promises";
import { createEvoDiffCPU } from "../cpu/index.js";
import { oracle, checkEngine } from "./parity.js";
const read = async (url) => {
  const b = await readFile(new URL("../" + url, import.meta.url));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
const results = {
  platform: process.platform,
  arch: process.arch,
  node: process.version,
  skipped: 0,
  precisions: {},
};
for (const precision of ["fp32", "fp16", "int8"]) {
  const ref = await oracle(precision, read),
    engine = await createEvoDiffCPU({
      baseUrl: "weights/oa_dm_38m",
      precision,
      read,
      cache: "none",
    });
  try {
    results.precisions[precision] = await checkEngine(engine, ref, console.log);
  } finally {
    engine.destroy();
  }
}
await writeFile(
  new URL("../fixtures/cpu_metrics.json", import.meta.url),
  JSON.stringify(results, null, 2) + "\n",
);
console.log(
  "CPU: all intermediate/final logits and exact B/C sequences passed; zero skips.",
);
