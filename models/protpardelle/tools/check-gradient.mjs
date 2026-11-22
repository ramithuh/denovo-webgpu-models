// Optional sensitivity diagnostic, separate from the sampler's strict trace gates.
import { readFile, writeFile } from "node:fs/promises";
import { createProtpardelleCPU } from "../cpu/index.js";
import { root, read, arrays, inputFrom, metrics } from "../test/helpers.mjs";
import { oracleBinary } from "../test/fixtures.js";

const records = [];
for (const variant of ["cc58_epoch416", "cc78_epoch1431", "cc91_epoch383"]) {
  const meta = JSON.parse(
    await readFile(root + `fixtures/oracle_gradient_${variant}.json`, "utf8"),
  );
  const binary = await oracleBinary(meta, root + "fixtures/", read);
  for (const entry of meta.cases) {
    const engine = await createProtpardelleCPU({
      baseUrl: root + `weights/${variant}/${entry.precision}`,
      read,
    });
    try {
      const data = arrays(entry.inputs, binary);
      const expected = arrays(entry.traces, binary);
      const result = engine.gradient(inputFrom(data), {
        target: data.target,
        lossMask: data.loss_mask,
      });
      const gradient = metrics(result.gradient, expected.gradient);
      records.push({
        variant,
        precision: entry.precision,
        gradient,
        coordinates: metrics(result.coords, expected.denoised_coords),
        belowStrictTraceTolerance: gradient.relRMS < 1e-4,
      });
    } finally {
      engine.destroy();
    }
  }
}
await writeFile(
  root + "fixtures/gradient_diagnostic_metrics.json",
  JSON.stringify(
    {
      scope:
        "CPU coordinate Jacobian on dense random atom37 clouds; this is a sensitivity diagnostic, not a passing release gate",
      tolerance: 1e-4,
      records,
    },
    null,
    2,
  ) + "\n",
);
const failures = records.filter((record) => !record.belowStrictTraceTolerance);
console.log(
  `${failures.length}/${records.length} dense-cloud gradient diagnostics exceed 1e-4; see fixtures/gradient_diagnostic_metrics.json`,
);
if (failures.length) process.exitCode = 1;
