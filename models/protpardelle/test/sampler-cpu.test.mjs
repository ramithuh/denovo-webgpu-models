import test from "node:test";
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { createProtpardelleCPU } from "../cpu/index.js";
import { root, read, metrics } from "./helpers.mjs";
import { samplerOptions, miniInput } from "./sampler-fixture.js";
test(
  "complete sampler replay matches every upstream running preset",
  { timeout: 3600000 },
  async () => {
    const inventory = JSON.parse(
        await readFile(root + "fixtures/sampler_inventory.json", "utf8"),
      ),
      pdb = await readFile(root + "fixtures/sampler_motif.pdb", "utf8"),
      results = [];
    assert.equal(inventory.length, 111);
    for (const entry of inventory) {
      if (process.env.PORT_PRESET && entry.preset !== process.env.PORT_PRESET)
        continue;
      assert.equal(entry.upstream_status, "executed", entry.preset);
      const meta = JSON.parse(
          await readFile(
            root + `fixtures/sampler_${entry.preset}.json`,
            "utf8",
          ),
        ),
        binary = await read(root + `fixtures/sampler_${entry.preset}.bin`),
        { options, record } = samplerOptions(meta, binary, pdb);
      const engine = await createProtpardelleCPU({
        baseUrl: root + `weights/${entry.variant}/${meta.precision}`,
        read,
      });
      try {
        let maxTraceRelRMS = 0,
          traceCount = 0;
        const out = await engine.sample(options, { returnIntermediates: true });
        const m = metrics(
          out.coords,
          (record.stage2_outputs ?? record.outputs).x,
        );
        console.log(entry.preset, m);
        assert.ok(m.relRMS < 1e-4, `${entry.preset}: ${JSON.stringify(m)}`);
        assert.deepEqual(Array.from(out.aatype), Array.from(record.outputs.s));
        assert.deepEqual(
          Array.from(out.atomMask),
          Array.from((record.stage2_outputs ?? record.outputs).atom_mask),
        );
        for (const [prefix, calls] of [
          ["", record.denoiser_calls],
          ["stage2.", record.stage2_denoiser_calls ?? []],
        ])
          for (let step = 0; step < calls.length; step++) {
            const mini = miniInput(calls[step]);
            const isolated = mini
              ? engine.sequence(mini, { returnIntermediates: true })
              : null;
            for (const [name, expected] of Object.entries(calls[step].traces)) {
              const actual = name.startsWith("mpnn.")
                ? isolated.intermediates[name]
                : out.intermediates[
                    `${prefix}step.${Math.floor(step / (options.conditioning.recurrence ?? 1))}.${step % (options.conditioning.recurrence ?? 1)}.${name}`
                  ];
              assert.ok(actual, `Missing ${name}`);
              const m = metrics(actual, expected);
              maxTraceRelRMS = Math.max(maxTraceRelRMS, m.relRMS);
              traceCount++;
              assert.ok(
                m.relRMS < 1e-4,
                `${entry.preset}/${prefix}${step}/${name}: ${m.relRMS}`,
              );
            }
          }
        for (const [prefix, expected] of [
          ["", record.outputs],
          ["stage2.", record.stage2_outputs],
        ])
          if (expected)
            for (let step = 0; step < expected.xt_traj.length; step++) {
              const metric = metrics(
                out.intermediates[`${prefix}trajectory.${step}.coords`],
                expected.xt_traj[step],
              );
              assert.ok(
                metric.relRMS < 1e-4,
                `${entry.preset}/${prefix}trajectory.${step}: ${metric.relRMS}`,
              );
              assert.deepEqual(
                Array.from(
                  out.intermediates[`${prefix}trajectory.${step}.sequence`],
                ),
                Array.from(expected.st_traj[step]),
              );
              maxTraceRelRMS = Math.max(maxTraceRelRMS, metric.relRMS);
              traceCount++;
            }
        results.push({
          preset: entry.preset,
          ...m,
          traceCount,
          maxTraceRelRMS,
          miniTraceBoundary: "recorded original MiniMPNN inputs",
        });
      } finally {
        engine.destroy();
      }
    }
    assert.ok(results.length);
    await writeFile(
      root +
        `fixtures/sampler_cpu_metrics${process.env.PORT_PRESET ? "_" + process.env.PORT_PRESET : ""}.json`,
      JSON.stringify(results, null, 2) + "\n",
    );
  },
);
