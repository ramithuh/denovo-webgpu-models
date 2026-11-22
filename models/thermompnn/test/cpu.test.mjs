import test from "node:test";
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { createThermoMPNNCPU } from "../cpu/index.js";
import { read, readJSON, oracle, relRMS } from "./helpers.mjs";

test("CPU traces, saturation and targeted point mutations: all modes and storage precisions", async () => {
  const cases = await readJSON("fixtures/cases.json"),
    metrics = {};
  for (const precision of ["fp32", "fp16", "int8"]) {
    const engine = await createThermoMPNNCPU({
      baseUrl: "weights/default",
      precision,
      read,
    });
    metrics[precision] = {};
    try {
      for (const c of cases) {
        const text = new TextDecoder().decode(
          await read("fixtures/" + c.input),
        );
        const options = {
          chains: c.inputChains ?? c.chains,
          returnIntermediates: true,
        };
        const result = await engine.predict(text, options),
          { traces } = await oracle(c, precision),
          m = {};
        for (const [name, expected] of Object.entries(traces)) {
          assert.ok(result.intermediates[name], `Missing trace ${name}`);
          if (expected instanceof Int32Array)
            assert.deepEqual(
              result.intermediates[name],
              expected,
              `${precision}/${c.id}/${name}`,
            );
          m[name] = relRMS(result.intermediates[name], expected);
          assert.ok(
            m[name].relRMS < 1e-4,
            `${precision}/${c.id}/${name}: ${JSON.stringify(m[name])}`,
          );
        }
        for (const r of result.residues)
          if (r.aa !== "X")
            assert.equal(
              result.ddg[r.index * 20 + result.alphabet.indexOf(r.aa)],
              0,
            );
        const r = result.residues.find((r) => r.aa !== "X");
        const targeted = await engine.predict(text, {
          chains: options.chains,
          mode: "targeted",
          mutations: [
            {
              chain: r.chain,
              number: r.number,
              ins: r.ins,
              wildtype: r.aa,
              mutation: "A",
            },
            { index: r.index, mutation: r.aa },
          ],
        });
        assert.equal(targeted.scores[0], result.ddg[r.index * 20]);
        assert.equal(targeted.scores[1], 0);
        metrics[precision][c.id] = { timingMs: result.timingMs, traces: m };
        console.log(
          `${precision}/${c.id}: ddg ${m.ddg.relRMS.toExponential(3)}, ${result.timingMs.toFixed(0)} ms`,
        );
      }
      const full = await engine.predict(
          new TextDecoder().decode(await read("fixtures/full.pdb")),
        ),
        ref = await oracle({ id: "full_length" }, precision);
      const m = relRMS(full.ddg, ref.traces.ddg);
      assert.ok(m.relRMS < 1e-4, `Full chain ${precision}`);
      metrics[precision].full_length = {
        timingMs: full.timingMs,
        traces: { ddg: m },
      };
    } finally {
      engine.destroy();
    }
  }
  await writeFile(
    new URL("../fixtures/cpu_metrics.json", import.meta.url),
    JSON.stringify(metrics, null, 2) + "\n",
  );
});

test("CPU engine rejects invalid modes, targets, missing weights, and cancellation", async () => {
  await assert.rejects(
    createThermoMPNNCPU({ baseUrl: "missing-weights", read }),
  );
  const engine = await createThermoMPNNCPU({
      baseUrl: "weights/default",
      read,
    }),
    text = new TextDecoder().decode(await read("fixtures/small.pdb"));
  try {
    await assert.rejects(engine.predict(""));
    await assert.rejects(engine.predict(text, { mode: "double" }));
    await assert.rejects(
      engine.predict(text, { mode: "targeted", mutations: [] }),
    );
    await assert.rejects(engine.predict(text, { chains: ["missing"] }));
    await assert.rejects(
      engine.predict(text, {
        mode: "targeted",
        mutations: [{ index: 0, mutation: "X" }],
      }),
    );
    const controller = new AbortController();
    await assert.rejects(
      engine.predict(text, {
        signal: controller.signal,
        onProgress: () => controller.abort(),
      }),
    );
  } finally {
    engine.destroy();
    engine.destroy();
  }
  await assert.rejects(engine.predict(text), /destroyed/);
  assert.equal(engine.memoryUsage().residentBytes, 0);
});
