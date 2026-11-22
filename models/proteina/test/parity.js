import { sha256 } from "../shared/weights.js";
export function assert(x, m) {
  if (!x) throw Error(m);
}
export function parity(a, b, label, tolerance = 1e-4) {
  assert(a?.length === b?.length, label + " shape");
  let error = 0,
    ref = 0,
    maxAbs = 0;
  for (let i = 0; i < a.length; i++) {
    assert(
      Number.isFinite(a[i]) && Number.isFinite(b[i]),
      label + " nonfinite",
    );
    const d = a[i] - b[i];
    error += d * d;
    ref += b[i] * b[i];
    maxAbs = Math.max(maxAbs, Math.abs(d));
  }
  const relRMS = Math.sqrt(error / Math.max(ref, 1e-30));
  assert(relRMS < tolerance, `${label}: relRMS=${relRMS}, maxAbs=${maxAbs}`);
  return { relRMS, maxAbs };
}
export async function oracle(precision, read) {
  const meta = JSON.parse(
      new TextDecoder().decode(await read(`fixtures/oracle_${precision}.json`)),
    ),
    data = await read(meta.binary);
  assert((await sha256(data)) === meta.sha256, "Oracle SHA-256");
  return {
    meta,
    tensor(name) {
      const t = meta.tensors[name];
      assert(t, "Missing tensor " + name);
      return new Float32Array(data, t.offset, t.length);
    },
  };
}
export async function checkEngine(engine, ref, log = () => {}, trace = false) {
  const results = { denoise: [], samples: [] };
  for (const c of ref.meta.denoise) {
    await log(c.id);
    const out = await engine.forward(ref.tensor(c.x), {
      t: c.t,
      xSC: c.xSC ? ref.tensor(c.xSC) : undefined,
      fold: c.fold ?? undefined,
      trace: trace && Object.keys(c.trace).length > 0,
    });
    if (trace)
      for (const [n, t] of Object.entries(c.trace)) {
        let actual = out.trace[n];
        const expected = ref.tensor(t);
        if (["cond_factory", "transition_c_1", "transition_c_2"].includes(n)) {
          const expanded = new Float32Array(expected.length);
          for (let i = 0; i < c.length; i++) expanded.set(actual, i * 512);
          actual = expanded;
        }
        parity(actual, expected, n);
      }
    results.denoise.push({
      id: c.id,
      ...parity(out.coordinates, ref.tensor(c.output), c.id),
      timingMs: out.timingMs,
    });
  }
  for (const c of ref.meta.samples) {
    await log(c.id);
    const out = await engine.sample({
      ...c,
      noise: c.noise.map((n) => ref.tensor(n)),
      trajectory: true,
    });
    for (let i = 0; i < c.steps; i++)
      parity(
        out.trajectory[i],
        ref.tensor(c.trajectory[i]),
        c.id + " step " + i,
      );
    results.samples.push({
      id: c.id,
      ...parity(out.coordinates, ref.tensor(c.output), c.id),
    });
  }
  return results;
}
