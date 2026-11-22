import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fixture, metrics } from "./common.js";
import { CPUAlgebra } from "../cpu/algebra.js";
import { integrate, vectorfields } from "../shared/geometry.js";
const read = async (p) => {
  const b = await readFile(new URL("../" + p, import.meta.url));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
test("SO(3) small-angle logarithm and upstream epsilon survive integration", async () => {
  const L = 64,
    T = 3,
    { v } = await fixture(read, "fp32", L),
    o = new CPUAlgebra();
  for (let step = 0; step < T; step++) {
    const trajectory = v["T3.trajectory"],
      f = o.tensor(
        [L, 7],
        step === 0
          ? v.initial
          : trajectory.subarray((step - 1) * L * 7, step * L * 7),
      ),
      p = o.tensor([L, 7], v[`T3.step${step}.pred`]),
      pred = { q: o.slice(p, 1, 0, 4), t: o.slice(p, 1, 4, 7) },
      time = 1 - step * 0.495;
    const actual = await o.read(integrate(o, f, pred, time, 0.495));
    assert.ok(
      metrics(actual, trajectory.subarray(step * L * 7, (step + 1) * L * 7))
        .relRMS < 2e-6,
    );
    if (step === 2) {
      const vf = await o.read(vectorfields(o, f, pred, time).rot);
      assert.ok(metrics(vf, v["T3.rotVF"]).relRMS < 2e-6);
    }
  }
});
