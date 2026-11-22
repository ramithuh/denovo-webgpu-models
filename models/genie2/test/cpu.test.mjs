import test from "node:test";
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { createGenie2CPUEngine } from "../cpu/index.js";
import { diffStats } from "../shared/weights.js";
import { fixture, read } from "./fixtures.mjs";
test(
  "Complete CPU trajectories against unmodified pinned PyTorch; FP32, all modes and lengths",
  { timeout: 1800000 },
  async (t) => {
    const records = [];
    for (const precision of ["fp32"]) {
      const e = await createGenie2CPUEngine({
        baseUrl: `weights/${precision}`,
        precision,
        read,
      });
      try {
        for (let i = 0; i < 6; i++)
          await t.test(`${precision} case ${i}`, async () => {
            const { c, get, features } = await fixture(precision, i),
              start = performance.now();
            const out = await e.sample({
              length: c.length,
              features,
              steps: c.steps,
              scale: c.scale,
              initialNoise: get("initial"),
              noises: Array.from({ length: c.steps }, (_, k) =>
                get(`step${c.steps - k}.noise`),
              ),
              trajectory: true,
            });
            const result = {
              case: c.name,
              final: diffStats(out.coordinates, get("final")),
              trajectory: out.trajectory.map((x, k) =>
                diffStats(x, get(`step${c.steps - k}.output`)),
              ),
              elapsedMs: performance.now() - start,
            };
            records.push(result);
            assert.equal(e.memoryUsage().transientBuffers, 0);
            assert.ok(result.final.relRms < 1e-4, JSON.stringify(result));
          });
      } finally {
        e.destroy();
        e.destroy();
        assert.equal(e.memoryUsage().bytes, 0);
      }
    }
    await writeFile(
      new URL("../fixtures/cpu_metrics.json", import.meta.url),
      JSON.stringify(
        {
          threshold: 1e-4,
          reference: "unmodified pinned PyTorch",
          records,
          skipped: 0,
          passed: records.every((r) => r.final.relRms < 1e-4),
        },
        null,
        2,
      ) + "\n",
    );
  },
);
