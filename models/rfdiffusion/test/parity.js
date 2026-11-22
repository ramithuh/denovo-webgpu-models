import { sha256 } from "../shared/loader.js";
export function metric(actual, expected) {
  if (!actual || actual.length !== expected.length)
    throw Error("Trace length mismatch");
  let error = 0,
    norm = 0,
    maxAbs = 0;
  for (let i = 0; i < actual.length; i++) {
    if (!Number.isFinite(actual[i]) || !Number.isFinite(expected[i]))
      throw Error(`Nonfinite value at ${i}`);
    const d = actual[i] - expected[i];
    error += d * d;
    norm += expected[i] * expected[i];
    maxAbs = Math.max(maxAbs, Math.abs(d));
  }
  return { relRMS: Math.sqrt(error / Math.max(norm, 1e-30)), maxAbs };
}
export async function readFixture(name, read) {
  const decode = new TextDecoder(),
    meta = JSON.parse(
      decode.decode(await read(`/fixtures/oracle_${name}.json`)),
    ),
    bin = await fixtureBinary(meta, read);
  const arrays = (entries) =>
    Object.fromEntries(
      Object.entries(entries).map(([k, v]) => [
        k,
        new Float32Array(bin.slice(v.offset, v.offset + v.length * 4)),
      ]),
    );
  return {
    meta,
    input: {
      length: meta.length,
      symmetry: meta.mode === "symmetric" ? "c3" : undefined,
      ...arrays(meta.inputs),
    },
    expected: arrays(meta.traces),
  };
}
export async function readTrajectory(name, read) {
  const meta = JSON.parse(
      new TextDecoder().decode(await read(`/fixtures/${name}.json`)),
    ),
    bin = await fixtureBinary(meta, read),
    array = (v) =>
      new Float32Array(bin.slice(v.offset, v.offset + v.length * 4));
  return {
    meta,
    noise: {
      initialCoords: array(meta.initial),
      sequence: array(meta.sequence),
      diffusionMask: array(meta.mask),
      steps: meta.steps.map((s) => ({
        translation: array(s.translation),
        rotation: array(s.rotationNoise),
      })),
    },
    expected: meta.steps.map((s) =>
      Object.fromEntries(
        ["px0", "xyz", "coords", "rotation"].map((k) => [k, array(s[k])]),
      ),
    ),
  };
}
export function check(actual, expected, name) {
  const result = metric(actual, expected);
  if (!(result.relRMS < 1e-4))
    throw Error(`${name}: relRMS=${result.relRMS}, maxAbs=${result.maxAbs}`);
  return result;
}
export async function verifyDenoiser(engine, fixture) {
  const out = await engine.denoiseStep(fixture.input, {
    returnIntermediates: true,
  });
  let worst = { name: "", relRMS: 0 },
    n = 0;
  for (const [name, expected] of Object.entries(fixture.expected)) {
    const m = check(
      out.intermediates[name],
      expected,
      fixture.meta.name + "/" + name,
    );
    if (m.relRMS > worst.relRMS) worst = { name, ...m };
    n++;
  }
  return {
    name: fixture.meta.name,
    traceCount: n,
    worst,
    px0: metric(out.px0, fixture.expected.px0),
    backbone: metric(out.coords, fixture.expected.backbone),
    timingMs: out.timingMs,
  };
}
export async function verifyTrajectory(engine, fixture) {
  const out = await engine.sample({
    length: fixture.meta.length,
    symmetry: fixture.meta.mode === "symmetric" ? "c3" : undefined,
    numSteps: 3,
    recordedNoise: fixture.noise,
  });
  const metrics = [];
  for (let j = 0; j < 3; j++)
    for (const [key, expected] of Object.entries(fixture.expected[j]))
      metrics.push({
        step: j,
        t: out.trajectory[j].t,
        key,
        ...check(
          out.trajectory[j][key],
          expected,
          fixture.meta.name + "/" + j + "/" + key,
        ),
      });
  return { name: fixture.meta.name, metrics, timingMs: out.timingMs };
}

async function fixtureBinary(meta, read) {
  const bin = await read("/fixtures/" + meta.binary);
  if (
    bin.byteLength !== meta.binary_byteLength ||
    (await sha256(bin)) !== meta.binary_sha256
  )
    throw Error("Oracle binary integrity failure: " + meta.name);
  return bin;
}
