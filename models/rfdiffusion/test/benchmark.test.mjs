import test from "node:test";
import { writeFile } from "node:fs/promises";
import { browserTask } from "./browser.mjs";
import { root } from "./helpers.mjs";
test(
  "NVIDIA GB10 cold-load and warm denoiser timing at L24 and L64",
  { timeout: 3600000 },
  async () => {
    const result = await browserTask((page) =>
      page.evaluate(async () => {
        const { createRFdiffusionEngine } = await import("/webgpu/index.js"),
          { prepareInput, gaussian, seededRandom } = await import(
            "/shared/preprocess.js"
          ),
          chemical = (await import("/shared/chemical-constants.js")).default,
          rows = [];
        for (const precision of ["fp32", "fp16", "int8"]) {
          const t = performance.now(),
            engine = await createRFdiffusionEngine({
              baseUrl: "/weights",
              precision,
              cache: "none",
            }),
            coldLoadMs = performance.now() - t;
          try {
            for (const length of [24, 64]) {
              const random = seededRandom(11),
                coords = new Float32Array(length * 9);
              for (let i = 0; i < length; i++) {
                const shift = [
                  gaussian(random) * 4,
                  gaussian(random) * 4,
                  gaussian(random) * 4,
                ];
                for (let a = 0; a < 3; a++)
                  coords.set(
                    chemical.initial[a].map((x, j) => x + shift[j]),
                    i * 9 + a * 3,
                  );
              }
              const state = prepareInput({ length, coords }),
                warm = [];
              await engine.denoiseStep(state);
              for (let i = 0; i < 3; i++)
                warm.push((await engine.denoiseStep(state)).timingMs);
              rows.push({
                precision,
                length,
                coldLoadMs,
                warmMs: warm,
                medianMs: warm.slice().sort((a, b) => a - b)[1],
              });
            }
          } finally {
            engine.destroy();
          }
        }
        return rows;
      }),
    );
    await writeFile(
      root + "fixtures/benchmarks.json",
      JSON.stringify(result, null, 2) + "\n",
    );
  },
);
