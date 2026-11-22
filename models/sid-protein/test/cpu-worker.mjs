import { parentPort, workerData } from "node:worker_threads";
import { createSidProteinCpuEngine } from "../cpu/index.js";
import { diffStats } from "../shared/index.js";
import { root, read, json, floats } from "./io.mjs";
const oracle = await json("fixtures/oracle_fp32.json"),
  engine = await createSidProteinCpuEngine({
    baseUrl: new URL("weights/fp32", root).href,
    read,
  });
const rows = [];
try {
  for (const index of workerData.indices) {
    const c = oracle.cases[index];
    const input = await floats(c.forwardInput),
      expected = await floats(c.forward),
      start = performance.now();
    const out = await engine.forward(input, 0.37, {
        cathIndices: c.cathIndices,
      }),
      forward = diffStats(out, expected);
    if (forward.relRms >= 1e-4)
      throw new Error(
        "CPU forward parity failed " +
          JSON.stringify({ name: c.name, forward }),
      );
    const ref = await floats(c.trajectory),
      result = await engine.generate({
        length: c.length,
        steps: c.steps,
        noiseScale: c.noiseScale,
        noise: await floats(c.noise),
        cathIndices: c.cathIndices,
        onStep: ({ step }) =>
          parentPort.postMessage({
            kind: "progress",
            name: c.name,
            step,
            steps: c.steps,
          }),
      });
    const sampling = diffStats(result.coordinates, ref.slice(-c.length * 3));
    const trajectory = diffStats(
      Float32Array.from(result.trajectory.flatMap((v) => Array.from(v))),
      ref,
    );
    const row = {
      name: c.name,
      forward,
      sampling,
      trajectory,
      milliseconds: performance.now() - start,
    };
    rows.push(row);
    parentPort.postMessage({ kind: "row", row });
    if (sampling.relRms >= 1e-4 || trajectory.relRms >= 1e-4)
      throw new Error("CPU sampling parity failed " + JSON.stringify(row));
  }
} finally {
  engine.destroy();
}
parentPort.postMessage({
  kind: "done",
  rows,
  memoryAfterDestroy: engine.memoryUsage(),
});
