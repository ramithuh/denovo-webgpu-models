import test from "node:test";
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { createBoltzGenCPUEngine } from "../cpu/index.js";
import {
  configurations,
  cases,
  casesFor,
  paths,
  verifyGenerated,
} from "./verify.js";
const root = resolve(import.meta.dirname, "..");
const read = async (path) => {
  const b = await readFile(path);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
for (const [variant, precision] of configurations)
  test(`complete CPU ${variant} ${precision}`, async () => {
    const engine = await createBoltzGenCPUEngine({
      baseUrl: `${root}/weights/${variant}/${precision}`,
      read,
    });
    try {
      for (const scenario of casesFor(variant, precision)) {
        const files = paths(root, variant, precision, scenario),
          start = performance.now(),
          result = await engine.generate({
            ...scenario,
            targetPDB: scenario.targetFile
              ? await readFile(`${root}/${scenario.targetFile}`, "utf8")
              : undefined,
          }),
          stats = await verifyGenerated(result, files, read);
        assert.equal(engine.memoryUsage().transientBytes, 0);
        console.log(
          "CPU PARITY",
          variant,
          precision,
          scenario.name,
          stats.atom_coords,
        );
        await writeFile(
          `${root}/fixtures/cpu_${variant}_${precision}_${scenario.name}.json`,
          JSON.stringify(
            {
              stats,
              seconds: (performance.now() - start) / 1000,
              timings: result.timings,
            },
            null,
            2,
          ) + "\n",
        );
      }
    } finally {
      engine.destroy();
    }
    assert.equal(engine.memoryUsage().totalBytes, 0);
  });
