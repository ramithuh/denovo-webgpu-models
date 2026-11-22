import test from "node:test";
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { createProtpardelleCPU } from "../cpu/index.js";
import { root, read, arrays, inputFrom, metrics } from "./helpers.mjs";
import { oracleBinary } from "./fixtures.js";
test(
  "CPU denoiser intermediate parity for every released structure checkpoint",
  { timeout: 3600000 },
  async () => {
    const matrix = JSON.parse(
      await readFile(root + "fixtures/mode_matrix.json", "utf8"),
    );
    assert.equal(
      matrix.checkpoints.length,
      16,
      "Verified checkpoint inventory required",
    );
    const result = [];
    for (const checkpoint of matrix.checkpoints) {
      if (
        process.env.PORT_VARIANT &&
        checkpoint.name !== process.env.PORT_VARIANT
      )
        continue;
      const meta = JSON.parse(
        await readFile(
          root + `fixtures/oracle_${checkpoint.name}.json`,
          "utf8",
        ),
      );
      const binary = await oracleBinary(meta, root + "fixtures/", read);
      for (const precision of ["fp32", "fp16", "int8"]) {
        const engine = await createProtpardelleCPU({
          baseUrl: root + `weights/${checkpoint.name}/${precision}`,
          read,
        });
        try {
          for (const entry of meta.cases.filter(
            (c) => c.precision === precision,
          )) {
            const input = inputFrom(arrays(entry.inputs, binary)),
              expected = arrays(entry.traces, binary);
            const out = engine[
              meta.scope === "minimpnn" ? "sequence" : "denoise"
            ](input, { returnIntermediates: true });
            for (const [name, values] of Object.entries(expected)) {
              assert.ok(out.intermediates[name], `Missing trace ${name}`);
              const m = metrics(out.intermediates[name], values);
              assert.ok(
                m.relRMS < 1e-4,
                `${checkpoint.name}/${precision}/${entry.mode}/${name}: ${JSON.stringify(m)}`,
              );
              result.push({
                variant: checkpoint.name,
                precision,
                mode: entry.mode,
                trace: name,
                ...m,
              });
            }
            assert.throws(
              () =>
                engine.denoise({ ...input, noiseLevel: Float32Array.of(-1) }),
              /Invalid|Negative/,
            );
          }
        } finally {
          engine.destroy();
          engine.destroy();
        }
        console.log(
          `${checkpoint.name}/${precision}: all intermediate traces pass`,
        );
      }
    }
    assert.ok(result.length, "No CPU cases executed");
    await writeFile(
      root +
        `fixtures/cpu_metrics${process.env.PORT_VARIANT ? "_" + process.env.PORT_VARIANT : ""}.json`,
      JSON.stringify(result, null, 2) + "\n",
    );
  },
);
