import test from "node:test";
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { createEsmcCPU } from "../cpu/index.js";
import { oracle, verifyProtein, verifyTokens } from "./parity.js";
const root = new URL("../", import.meta.url);
const read = async (url) => {
  const b = await readFile(url);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
async function fixture(name) {
  const meta = JSON.parse(
    await readFile(new URL(`fixtures/oracle_fp32_${name}.json`, root)),
  );
  return {
    meta,
    ref: oracle(
      meta,
      await read(new URL(`fixtures/oracle_fp32_${name}.bin`, root)),
    ),
  };
}
test(
  "CPU FP32: all layers and all positions across the complete mode matrix",
  { timeout: 86400000 },
  async () => {
    const start = performance.now();
    const engine = await createEsmcCPU({
      baseUrl: new URL("weights/esmc-300m", root).pathname,
      read,
    });
    const metrics = {};
    try {
      for (const name of ["trp_cage", "ubiquitin"]) {
        console.log("CPU", name);
        const { meta, ref } = await fixture(name);
        metrics[name] = await verifyProtein(engine, meta, ref, {
          log: (s) => console.log("CPU", name, s),
        });
      }
      const { meta, ref } = await fixture("tokens");
      metrics.tokens = await verifyTokens(engine, meta, ref);
      const natural = await engine.run("AC", { mode: "naturalness" }),
        masked = await engine.run("AC", { mode: "masked" });
      assert.deepEqual(natural.llr, masked.llr);
      assert.equal(natural.pseudoPerplexity, masked.pseudoPerplexity);
      for (const s of ["", "ac", "A B", {}, new Int32Array([0, 99, 2])])
        assert.throws(() => engine.forward(s));
      assert.throws(() => engine.forward("AC", { sequenceIds: [1] }));
      assert.throws(() => engine.forward("AC", { hiddenLayer: 30 }));
      const c = new AbortController();
      c.abort();
      assert.throws(() => engine.forward("AC", { signal: c.signal }), {
        name: "AbortError",
      });
    } finally {
      engine.destroy();
    }
    assert.throws(() => engine.forward("AC"), /destroyed/);
    metrics.totalMs = performance.now() - start;
    await writeFile(
      new URL("fixtures/cpu_metrics.json", root),
      JSON.stringify(metrics, null, 2) + "\n",
    );
  },
);
