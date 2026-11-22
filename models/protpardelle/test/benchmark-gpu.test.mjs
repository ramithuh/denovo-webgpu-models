import test from "node:test";
import { writeFile } from "node:fs/promises";
import { root } from "./helpers.mjs";
import { browserTask } from "./browser.mjs";
test(
  "hardware cold load and warm 64/128-residue benchmark",
  { timeout: 120000 },
  async () => {
    const records = [];
    for (const precision of ["fp32", "fp16", "int8"])
      records.push(
        await browserTask((page) =>
          page.evaluate(async (precision) => {
            const { createProtpardelleEngine } = await import(
                "/webgpu/index.js"
              ),
              { chainLayout } = await import("/shared/inputs.js");
            const start = performance.now(),
              engine = await createProtpardelleEngine({
                baseUrl: "/weights",
                variant: "cc58_epoch416",
                precision,
              }),
              loadMs = performance.now() - start,
              results = [];
            try {
              for (const length of [64, 128]) {
                const input = {
                  ...chainLayout([length]),
                  coords: Float32Array.from(
                    { length: length * 111 },
                    (_, i) => Math.sin(i * 0.7) * 12,
                  ),
                  noiseLevel: new Float32Array(length).fill(12),
                };
                await engine.denoise(input);
                const warm = [];
                for (let j = 0; j < 5; j++)
                  warm.push((await engine.denoise(input)).timingMs);
                if (engine.memoryUsage().transientBuffers)
                  throw new Error("Benchmark leaked");
                results.push({ length, warmMs: warm });
              }
            } finally {
              engine.destroy();
            }
            return { precision, loadMs, results };
          }, precision),
        ),
      );
    await writeFile(
      root + "fixtures/benchmark_metrics.json",
      JSON.stringify(records, null, 2) + "\n",
    );
  },
);
