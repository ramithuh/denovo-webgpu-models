import { parentPort, workerData } from "node:worker_threads";
import { createRFdiffusionEngine } from "../cpu/index.js";
import { root, localRead } from "./helpers.mjs";
import {
  readFixture,
  readTrajectory,
  verifyDenoiser,
  verifyTrajectory,
} from "./parity.js";
const { precision, matrix, stress } = workerData;
const read = (url) => localRead(root + url.replace(/^\//, ""));
const engine = await createRFdiffusionEngine({
    baseUrl: root + "weights",
    precision,
    read: localRead,
  }),
  records = [];
try {
  const cases = stress
    ? [{ name: "fp32_unconditional_L64" }]
    : matrix.cases.filter((c) => c.precision === precision);
  for (const c of cases) {
    records.push(await verifyDenoiser(engine, await readFixture(c.name, read)));
    parentPort.postMessage({ progress: c.name });
  }
  if (!stress)
    for (const name of matrix.trajectories.filter((n) =>
      n.includes("_" + precision + "_"),
    )) {
      records.push(
        await verifyTrajectory(engine, await readTrajectory(name, read)),
      );
      parentPort.postMessage({ progress: name });
    }
} finally {
  engine.destroy();
}
if (engine.memoryUsage().bytes !== 0)
  throw Error("CPU engine retained model weights after destroy");
parentPort.postMessage({ records });
