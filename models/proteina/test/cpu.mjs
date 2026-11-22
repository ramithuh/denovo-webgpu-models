import { readFile, writeFile } from "node:fs/promises";
import { createProteinaCPU } from "../cpu/index.js";
import { oracle, checkEngine } from "./parity.js";
const root = new URL("../", import.meta.url),
  read = async (url) => {
    const b = await readFile(new URL(url, root));
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  };
const ref = await oracle("fp32", read),
  engine = await createProteinaCPU({
    baseUrl: "weights",
    precision: "fp32",
    read,
    cache: "none",
  });
try {
  const results = await checkEngine(engine, ref, console.log, true);
  await writeFile(
    new URL("../fixtures/cpu_metrics.json", import.meta.url),
    JSON.stringify({ skipped: 0, ...results }, null, 2) + "\n",
  );
  console.log(JSON.stringify(results, null, 2));
} finally {
  engine.destroy();
}
