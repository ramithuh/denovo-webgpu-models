import { createFoldFlow2Engine } from "../webgpu/index.js";
import { acquire } from "../webgpu/device.js";
import { GPUAlgebra } from "../webgpu/algebra.js";
import { fixture, metrics, check, firstAtoms } from "./common.js";
import { storageTests } from "./storage.js";
import { workerTest } from "./worker.js";
const log = async (...x) =>
  fetch("/log", { method: "POST", body: JSON.stringify(x) });
const read = async (p) => {
  const r = await fetch("/" + p);
  if (!r.ok) throw Error("Missing " + p);
  return r.arrayBuffer();
};
async function mustReject(fn, match) {
  let caught = false;
  try {
    await fn();
  } catch (e) {
    if (!match.test(String(e))) throw e;
    caught = true;
  }
  if (!caught) throw Error("Expected rejection " + match);
}
export async function run() {
  const rows = [],
    memory = [],
    timings = [],
    drift = [],
    lifecycle = [];
  let adapter, workerFixture;
  const storage = await storageTests();
  await log("storage", storage);
  for (const precision of ["fp32", "fp16", "int8"]) {
    const loadStart = performance.now(),
      engine = await createFoldFlow2Engine({
        baseUrl: "/weights/" + precision,
        precision,
        allowSoftwareFallback: false,
      });
    adapter = engine.adapterInfo;
    memory.push({ precision, ...engine.memoryUsage() });
    timings.push({
      precision,
      phase: "load",
      ms: performance.now() - loadStart,
    });
    await log("loaded", precision, adapter, engine.memoryUsage());
    try {
      for (const L of [32, 64]) {
        const { v } = await fixture(read, precision, L),
          opts = { length: L, initial: v.initial, trace: true };
        const start = performance.now(),
          r = await engine.forward(opts);
        timings.push({
          precision,
          L,
          phase: "cold_full_forward",
          ms: performance.now() - start,
        });
        for (const [name, a, b] of [
          ["esm.single", r.trace["esm.single"], v["esm.single"]],
          ["esm.pair", r.trace["esm.pair"], v["esm.pair"]],
          ["rigids", r.rigids, v["forward.rigids"]],
          ["psi", r.psi, v["forward.psi"]],
          ["atoms", r.atoms, firstAtoms(v["forward.atom37"])],
          ["atom37", r.atom37, v["forward.atom37"]],
          ["atom14", r.atom14, v["forward.atom14"]],
          ["rotVF", r.rotVectorfield, v["forward.rot_vectorfield"]],
          ["transVF", r.transVectorfield, v["forward.trans_vectorfield"]],
        ])
          rows.push(check(`${precision}/${L}/${name}`, a, b));
        await log(precision, L, "full forward", rows.slice(-5));
        const warmStart = performance.now(),
          warm = await engine.forward({ ...opts, trace: false });
        timings.push({
          precision,
          L,
          phase: "warm_full_forward",
          ms: performance.now() - warmStart,
        });
        rows.push(
          check(`${precision}/${L}/warm`, warm.rigids, v["forward.rigids"]),
        );
        // Keep raw representations for exercising the optional precomputed-input and abort APIs.
        const seqEmbed = {
          single: r.trace["esm.single"],
          pair: r.trace["esm.pair"],
        };
        for (const steps of [2, 3]) {
          const ts = performance.now(),
            sample = await engine.sample({
              ...opts,
              trace: false,
              steps,
            });
          timings.push({
            precision,
            L,
            steps,
            phase: "full_sample",
            ms: performance.now() - ts,
          });
          rows.push(
            check(
              `${precision}/${L}/T${steps}/trajectory`,
              Float32Array.from(
                sample.trajectory.flatMap((a) => Array.from(a)),
              ),
              v[`T${steps}.trajectory`],
            ),
          );
          rows.push(
            check(
              `${precision}/${L}/T${steps}/rotVF`,
              sample.rotVectorfield,
              v[`T${steps}.rotVF`],
            ),
          );
          rows.push(
            check(
              `${precision}/${L}/T${steps}/transVF`,
              sample.transVectorfield,
              v[`T${steps}.transVF`],
            ),
          );
          rows.push(
            check(
              `${precision}/${L}/T${steps}/pred`,
              sample.rigids,
              v[`T${steps}.step${steps - 1}.pred`],
            ),
          );
          rows.push(
            check(
              `${precision}/${L}/T${steps}/atoms`,
              sample.atoms,
              firstAtoms(v[`T${steps}.atoms`]),
            ),
          );
          for (const [name, actual, expected] of [
            ["psi", sample.psi, v[`T${steps}.psi`]],
            ["atom37", sample.atom37, v[`T${steps}.atoms`]],
          ])
            rows.push(check(`${precision}/${L}/T${steps}/${name}`, actual, expected));
        }
        if (precision !== "fp32") {
          const ref = await fixture(read, "fp32", L);
          drift.push({
            precision,
            L,
            frames: metrics(v["forward.rigids"], ref.v["forward.rigids"]),
            atoms: metrics(v["forward.atom37"], ref.v["forward.atom37"]),
            T2: metrics(v["T2.trajectory"], ref.v["T2.trajectory"]),
            T3: metrics(v["T3.trajectory"], ref.v["T3.trajectory"]),
          });
        }
        if (engine.memoryUsage().transientBuffers !== 0)
          throw Error("Leaked GPU transient");
        if (precision === "fp32" && L === 32) {
          for (const mode of ["precomputed", "tokens"]) {
            const conditioning =
              mode === "precomputed"
                ? {
                    seqEmbed: {
                      single: v["conditioned.single"],
                      pair: v["conditioned.pair"],
                    },
                  }
                : { esmTokens: Array.from(v["conditioned.tokens"]) };
            const conditioned = await engine.forward({
              length: L,
              initial: v.initial,
              ...conditioning,
            });
            rows.push(
              check(
                `conditional/${mode}/frames`,
                conditioned.rigids,
                v["conditioned.rigids"],
              ),
            );
            rows.push(
              check(
                `conditional/${mode}/atom37`,
                conditioned.atom37,
                v["conditioned.atom37"],
              ),
            );
          }
        }
        if (L === 32) {
          await mustReject(
            () =>
              engine.forward({
                ...opts,
                seqEmbed: {
                  single: new Float32Array(1),
                  pair: new Float32Array(1),
                },
              }),
            /shape/,
          );
          if (engine.memoryUsage().transientBuffers !== 0)
            throw Error("Error leaked transient");
          const c = new AbortController();
          await mustReject(
            () =>
              engine.sample({
                ...opts,
                seqEmbed,
                steps: 2,
                signal: c.signal,
                onProgress: () => c.abort(),
              }),
            /Abort/,
          );
          if (engine.memoryUsage().transientBuffers !== 0)
            throw Error("Abort leaked transient");
          lifecycle.push({ precision, normal: 0, error: 0, abort: 0 });
          if (precision === "int8") workerFixture = v;
        }
      }
    } finally {
      engine.destroy();
      engine.destroy();
    }
    if (engine.memoryUsage().bytes !== 0) throw Error("GPU destroy leak");
    lifecycle.at(-1).destroyBytes = engine.memoryUsage().bytes;
  }
  if (
    Math.abs(memory[1].bytes / memory[0].bytes - 0.5) > 0.01 ||
    Math.abs(memory[2].bytes / memory[0].bytes - 0.25) > 0.01
  )
    throw Error("Weights were not packed");
  const worker = await workerTest(workerFixture);
  rows.push(
    check(
      "worker/int8/L32/T2/atoms",
      worker.out.atoms,
      firstAtoms(workerFixture["T2.atoms"]),
    ),
  );
  // External-device ownership and admission use the exact production path on a tiny bundle.
  const env = await acquire({ allowSoftwareFallback: false }),
    external = await createFoldFlow2Engine({
      device: env.device,
      baseUrl: "/.cache/fixtures/loader",
    });
  external.destroy();
  const b = env.device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_DST,
  });
  b.destroy();
  const algebra = new GPUAlgebra(env.device);
  await mustReject(
    () =>
      Promise.resolve(
        algebra.alloc([Math.floor(env.device.limits.maxBufferSize / 4) + 1]),
      ),
    /exceeds/,
  );
  algebra.destroy();
  env.device.destroy();
  return {
    rows,
    memory,
    timings,
    drift,
    lifecycle,
    storage,
    worker: {
      initializationAbort: worker.initializationAbort,
      adapterInfo: worker.info.adapterInfo,
      memory: worker.memory,
      destroyed: worker.destroyed,
    },
    externalDevicePreserved: true,
    maxBufferSizeEnforced: true,
    adapter,
    skipped: 0,
  };
}
