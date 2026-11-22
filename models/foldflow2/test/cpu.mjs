import { readFile, writeFile } from "node:fs/promises";
import { createFoldFlow2CPU } from "../cpu/index.js";
import { fixture, check, firstAtoms } from "./common.js";
const root = new URL("../", import.meta.url),
  read = async (p) => {
    const b = await readFile(new URL(p, root));
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  };
const engine = await createFoldFlow2CPU({ baseUrl: "weights/fp32", read }),
  rows = [],
  timings = [];
try {
  for (const L of [32, 64]) {
    const { v } = await fixture(read, "fp32", L),
      opts = { length: L, initial: v.initial, trace: true },
      start = performance.now(),
      out = await engine.forward(opts);
    timings.push({ L, phase: "full_forward", ms: performance.now() - start });
    for (const [name, a, b] of [
      ["esm.single", out.trace["esm.single"], v["esm.single"]],
      ["esm.pair", out.trace["esm.pair"], v["esm.pair"]],
      ["rigids", out.rigids, v["forward.rigids"]],
      ["psi", out.psi, v["forward.psi"]],
      ["atoms", out.atoms, firstAtoms(v["forward.atom37"])],
      ["atom37", out.atom37, v["forward.atom37"]],
      ["atom14", out.atom14, v["forward.atom14"]],
      ["rotation_velocity", out.rotVectorfield, v["forward.rot_vectorfield"]],
      [
        "translation_velocity",
        out.transVectorfield,
        v["forward.trans_vectorfield"],
      ],
    ])
      rows.push(check(`${L}/${name}`, a, b));
    console.log("CPU full forward", L, rows.slice(-5));
    const seqEmbed = {
      single: out.trace["esm.single"],
      pair: out.trace["esm.pair"],
    };
    for (const steps of [2, 3]) {
      const ts = performance.now(),
        r = await engine.sample({ ...opts, seqEmbed, trace: false, steps });
      timings.push({
        L,
        steps,
        phase: "sample_cached_esm",
        ms: performance.now() - ts,
      });
      rows.push(
        check(
          `${L}/T${steps}/trajectory`,
          Float32Array.from(r.trajectory.flatMap((a) => Array.from(a))),
          v[`T${steps}.trajectory`],
        ),
      );
      rows.push(
        check(`${L}/T${steps}/rotVF`, r.rotVectorfield, v[`T${steps}.rotVF`]),
      );
      rows.push(
        check(
          `${L}/T${steps}/transVF`,
          r.transVectorfield,
          v[`T${steps}.transVF`],
        ),
      );
      rows.push(
        check(
          `${L}/T${steps}/atoms`,
          r.atoms,
          firstAtoms(v[`T${steps}.atoms`]),
        ),
      );
      for (const [name, actual, expected] of [
        ["pred", r.rigids, v[`T${steps}.step${steps - 1}.pred`]],
        ["psi", r.psi, v[`T${steps}.psi`]],
        ["atom37", r.atom37, v[`T${steps}.atoms`]],
      ])
        rows.push(check(`${L}/T${steps}/${name}`, actual, expected));
      console.log("CPU trajectory", L, steps, rows.slice(-3));
    }
    if (L === 32) {
      const conditioned = await engine.forward({
        length: L,
        initial: v.initial,
        seqEmbed: {
          single: v["conditioned.single"],
          pair: v["conditioned.pair"],
        },
      });
      rows.push(
        check(
          "conditional/precomputed/frames",
          conditioned.rigids,
          v["conditioned.rigids"],
        ),
      );
      rows.push(
        check(
          "conditional/precomputed/atom37",
          conditioned.atom37,
          v["conditioned.atom37"],
        ),
      );
    }
    if (engine.memoryUsage().transientBuffers !== 0)
      throw Error("CPU transient leak");
  }
} finally {
  engine.destroy();
  engine.destroy();
}
if (engine.memoryUsage().bytes !== 0) throw Error("CPU destroy leak");
await writeFile(
  new URL("../fixtures/cpu_metrics.json", import.meta.url),
  JSON.stringify({ rows, timings, skipped: 0 }, null, 2),
);
console.log("CPU parity passed:", rows.length, "checks, zero skips");
