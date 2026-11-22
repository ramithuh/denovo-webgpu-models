import { run as numericContracts } from "./numeric-suite.js";
import {
  lifecycle,
  cacheContracts,
  externalContracts,
  workerContracts,
  portableHalfContract,
  sensitivityRegression,
} from "./contracts.js";
import { createSidProteinEngine } from "../webgpu/index.js";
import { diffStats } from "../shared/index.js";
const log = async (x) =>
  fetch("/log", { method: "POST", body: JSON.stringify(x) });
const json = async (url) => {
  const r = await fetch(url);
  if (!r.ok) throw new Error("Missing fixture " + url);
  return r.json();
};
const floats = async (rec) => {
  const r = await fetch("/" + rec.file);
  if (!r.ok) throw new Error("Missing fixture " + rec.file);
  const data = await r.arrayBuffer();
  const hash = Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", data)),
    (v) => v.toString(16).padStart(2, "0"),
  ).join("");
  if (data.byteLength !== rec.byteLength || hash !== rec.sha256)
    throw new Error("Fixture checksum mismatch " + rec.file);
  return new Float32Array(data);
};
export async function run() {
  await log({ phase: "suite-start" });
  const rows = [],
    storage = {},
    contracts = {};
  let adapterInfo;
  contracts.numeric = await numericContracts();
  for (const precision of ["fp32", "fp16", "int8"]) {
    const engine = await createSidProteinEngine({
      baseUrl: `/weights/${precision}`,
      precision,
      allowSoftwareFallback: false,
      onProgress: (e) => {
        if (e.phase === "weights")
          log({ phase: "weights", precision, loaded: e.loaded });
      },
    });
    adapterInfo = engine.adapterInfo;
    await log({ precision, adapterInfo });
    storage[precision] = engine.memoryUsage();
    try {
      const oracle = await json(`/fixtures/oracle_${precision}.json`);

      for (const c of oracle.cases) {
        const input = await floats(c.forwardInput),
          ref = await floats(c.forward),
          start = performance.now();
        const actual = await engine.forward(input, 0.37, {
          cathIndices: c.cathIndices,
        });
        const forwardColdMs = performance.now() - start,
          forward = diffStats(actual, ref);
        await log({ precision, name: c.name, forward });
        if (forward.relRms >= 1e-4)
          throw new Error(
            "Forward parity failed " +
              JSON.stringify({ precision, name: c.name, forward }),
          );
        const warmStart = performance.now();
        await engine.forward(input, 0.37, { cathIndices: c.cathIndices });
        const warmMs = performance.now() - warmStart;
        const trajectory = await floats(c.trajectory),
          sstart = performance.now(),
          result = await engine.generate({
            length: c.length,
            steps: c.steps,
            noiseScale: c.noiseScale,
            cathIndices: c.cathIndices,
            noise: await floats(c.noise),
          });
        const sampling = diffStats(
          result.coordinates,
          trajectory.slice(-c.length * 3),
        );
        const all = diffStats(
          Float32Array.from(result.trajectory.flatMap((a) => Array.from(a))),
          trajectory,
        );
        const row = {
          precision,
          name: c.name,
          forward,
          sampling,
          trajectory: all,
          quantizationDrift: c.quantizationDrift,
          forwardColdMs,
          forwardWarmMs: warmMs,
          samplingMs: performance.now() - sstart,
          memoryAfterInference: engine.memoryUsage(),
        };
        rows.push(row);
        await log(row);
        if (sampling.relRms >= 1e-4 || all.relRms >= 1e-4)
          throw new Error("Sampling parity failed " + JSON.stringify(row));
        if (engine.memoryUsage().transientBuffers !== 0)
          throw new Error("Transient leak");
      }
      if (precision === "fp32") {
        contracts.lifecycle = await lifecycle(engine, oracle.cases[0], floats);
        contracts.sensitivityRegression = await sensitivityRegression(
          engine,
          floats,
        );
      }
    } finally {
      engine.destroy();
      if (engine.memoryUsage().totalBytes !== 0)
        throw new Error("Destroy leaked GPU buffers");
    }
  }
  const f16ratio = storage.fp16.weightBytes / storage.fp32.weightBytes,
    i8ratio = storage.int8.weightBytes / storage.fp32.weightBytes;
  if (f16ratio > 0.51 || i8ratio > 0.26)
    throw new Error("Weights were unpacked in VRAM");
  contracts.cache = await cacheContracts();
  const int8 = await json("/fixtures/oracle_int8.json");
  contracts.external = await externalContracts(int8.cases[0], floats);
  contracts.worker = await workerContracts(int8.cases[0], floats);
  const half = await json("/fixtures/oracle_fp16.json");
  contracts.portableHalf = await portableHalfContract(half.cases[0], floats);
  return {
    skipped: 0,
    adapterInfo,
    rows,
    storage,
    contracts,
    ratios: { fp16: f16ratio, int8: i8ratio },
  };
}
