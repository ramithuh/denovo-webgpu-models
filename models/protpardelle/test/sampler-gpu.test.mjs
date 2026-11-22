import test from "node:test";
import { readFile, writeFile } from "node:fs/promises";
import { browserTask } from "./browser.mjs";
import { root } from "./helpers.mjs";
test(
  "hardware sampler replays original trajectories and sequences",
  { timeout: 3600000 },
  async () => {
    const inventory = JSON.parse(
        await readFile(root + "fixtures/sampler_inventory.json", "utf8"),
      ),
      results = [];
    for (const entry of inventory) {
      if (process.env.PORT_PRESET && entry.preset !== process.env.PORT_PRESET)
        continue;
      const result = await browserTask(async (page) =>
        page.evaluate(async (entry) => {
          const { createProtpardelleEngine } = await import("/webgpu/index.js"),
            { samplerOptions, miniInput } = await import(
              "/test/sampler-fixture.js"
            );
          const meta = await (
              await fetch(`/fixtures/sampler_${entry.preset}.json`)
            ).json(),
            binary = await (
              await fetch(`/fixtures/sampler_${entry.preset}.bin`)
            ).arrayBuffer(),
            pdb = await (await fetch("/fixtures/sampler_motif.pdb")).text();
          const { options, record } = samplerOptions(meta, binary, pdb),
            engine = await createProtpardelleEngine({
              baseUrl: "/weights",
              variant: entry.variant,
              precision: meta.precision,
            });
          let maxTraceRelRMS = 0,
            traceCount = 0;
          const metrics = (a, b, label) => {
            if (!a || a.length !== b.length)
              throw new Error(`Missing ${label}`);
            let e = 0,
              n = 0,
              maxAbs = 0;
            for (let i = 0; i < a.length; i++) {
              const d = a[i] - b[i];
              e += d * d;
              n += b[i] * b[i];
              maxAbs = Math.max(maxAbs, Math.abs(d));
            }
            const relRMS = Math.sqrt(e / Math.max(n, 1e-30));
            maxTraceRelRMS = Math.max(maxTraceRelRMS, relRMS);
            traceCount++;
            if (!(relRMS < 1e-4))
              throw new Error(`${entry.preset}/${label}: ${relRMS}`);
            return { relRMS, maxAbs };
          };
          try {
            const out = await engine.sample(options, {
                returnIntermediates: true,
              }),
              m = metrics(
                out.coords,
                (record.stage2_outputs ?? record.outputs).x,
                "final",
              );
            if (
              Array.from(out.aatype).join() !==
              Array.from(record.outputs.s).join()
            )
              throw new Error("Sequence mismatch");
            metrics(
              out.atomMask,
              (record.stage2_outputs ?? record.outputs).atom_mask,
              "atom mask",
            );
            for (const [prefix, calls] of [
              ["", record.denoiser_calls],
              ["stage2.", record.stage2_denoiser_calls ?? []],
            ])
              for (let step = 0; step < calls.length; step++) {
                const mini = miniInput(calls[step]);
                const isolated = mini
                  ? await engine.sequence(mini, { returnIntermediates: true })
                  : null;
                for (const [name, expected] of Object.entries(
                  calls[step].traces,
                ))
                  metrics(
                    name.startsWith("mpnn.")
                      ? isolated.intermediates[name]
                      : out.intermediates[
                          `${prefix}step.${Math.floor(step / (options.conditioning.recurrence ?? 1))}.${step % (options.conditioning.recurrence ?? 1)}.${name}`
                        ],
                    expected,
                    `${prefix}${step}/${name}`,
                  );
              }
            for (const [prefix, expected] of [
              ["", record.outputs],
              ["stage2.", record.stage2_outputs],
            ])
              if (expected)
                for (let step = 0; step < expected.xt_traj.length; step++) {
                  metrics(
                    out.intermediates[`${prefix}trajectory.${step}.coords`],
                    expected.xt_traj[step],
                    `${prefix}trajectory.${step}`,
                  );
                  if (
                    Array.from(
                      out.intermediates[`${prefix}trajectory.${step}.sequence`],
                    ).join() !== Array.from(expected.st_traj[step]).join()
                  )
                    throw new Error("Trajectory sequence mismatch");
                }
            if (engine.memoryUsage().transientBuffers)
              throw new Error("Sampler buffer leak");
            return {
              preset: entry.preset,
              ...m,
              timingMs: out.timingMs,
              traceCount,
              maxTraceRelRMS,
              miniTraceBoundary: "recorded original MiniMPNN inputs",
            };
          } finally {
            engine.destroy();
            if (engine.memoryUsage().bytes) throw new Error("Resident leak");
          }
        }, entry),
      );
      results.push(result);
      console.log(result);
    }
    await writeFile(
      root +
        `fixtures/sampler_gpu_metrics${process.env.PORT_PRESET ? "_" + process.env.PORT_PRESET : ""}.json`,
      JSON.stringify(results, null, 2) + "\n",
    );
  },
);
