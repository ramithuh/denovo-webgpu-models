import test from "node:test";
import { writeFile } from "node:fs/promises";
import { browserTask } from "./browser.mjs";
import { root } from "./helpers.mjs";
test(
  "L64 end-to-end WGSL coordinate and block parity for every precision",
  { timeout: 3600000 },
  async () => {
    const records = await browserTask((page) =>
      page.evaluate(async () => {
        const { createRFdiffusionEngine } = await import("/webgpu/index.js"),
          { readFixture, verifyDenoiser } = await import("/test/parity.js"),
          read = async (u) => {
            const r = await fetch(u);
            if (!r.ok) throw Error("Missing " + u);
            return r.arrayBuffer();
          },
          records = [];
        for (const precision of ["fp32", "fp16", "int8"]) {
          const e = await createRFdiffusionEngine({
            baseUrl: "/weights",
            precision,
            cache: "none",
          });
          try {
            records.push(
              await verifyDenoiser(
                e,
                await readFixture(precision + "_unconditional_L64", read),
              ),
            );
            if (e.memoryUsage().transientBuffers !== 0)
              throw Error("L64 transient leak");
          } finally {
            e.destroy();
          }
          if (e.memoryUsage().bytes !== 0) throw Error("L64 destruction leak");
        }
        return records;
      }),
    );
    await writeFile(
      root + "fixtures/length64_metrics.json",
      JSON.stringify(records, null, 2) + "\n",
    );
  },
);
