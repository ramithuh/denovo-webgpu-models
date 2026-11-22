import { fixture, read } from "./fixtures.mjs";
import { createGenie2CPUEngine } from "../cpu/index.js";
import { diffStats } from "../shared/weights.js";
import { writeFile } from "node:fs/promises";
const { c, get, features } = await fixture("fp32");
const engine = await createGenie2CPUEngine({ baseUrl: "weights/fp32", read });
const rows = {};
const start = performance.now();
try {
  const r = await engine.forward({
    coordinates: get("initial"),
    rotations: get("step2.input_rots"),
    features,
    timestep: 2,
    steps: 2,
    onStage: async (n, t) => {
      const k = "step2." + n;
      if (c.tensors[k]) {
        const m = diffStats(t, get(k));
        rows[n] = m;
        console.log(n, JSON.stringify(m));
      }
    },
  });
  rows.elapsed = performance.now() - start;
  await writeFile(
    new URL("../.cache/cpu-diagnostic.json", import.meta.url),
    JSON.stringify(rows, null, 2),
  );
} finally {
  engine.destroy();
}
