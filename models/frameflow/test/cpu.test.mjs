import test from "node:test";
import { writeFile } from "node:fs/promises";
import { createFrameFlowCPU } from "../cpu/index.js";
import { fixture, verify } from "./parity.js";
import { root, read } from "./helpers.mjs";
test(
  "complete CPU network and dynamic trajectories against pinned PyTorch",
  { timeout: 3600000 },
  async () => {
    const records = [];
    const matrix = JSON.parse(
      new TextDecoder().decode(await read(root + "fixtures/matrix.json")),
    );
    matrix.cases.push(
      ...JSON.parse(
        new TextDecoder().decode(
          await read(root + "fixtures/motif_matrix.json"),
        ),
      ).cases,
    );
    for (const precision of ["fp32"]) {
      const engine = await createFrameFlowCPU({
        baseUrl: root + "weights",
        precision,
        read,
      });
      try {
        for (const name of matrix.cases.filter((n) =>
          n.startsWith(precision + "_"),
        )) {
          const r = await verify(engine, await fixture(name, read, root));
          records.push(r);
          console.log(name, r.ms);
        }
      } finally {
        engine.destroy();
      }
    }
    await writeFile(
      root + "fixtures/cpu_metrics.json",
      JSON.stringify(records, null, 2) + "\n",
    );
  },
);
