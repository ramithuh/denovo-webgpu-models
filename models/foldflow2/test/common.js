import { sha256 } from "../shared/loader.js";
export function metrics(a, b) {
  if (a.length !== b.length) throw Error(`shape ${a.length} != ${b.length}`);
  let s = 0,
    r = 0,
    max = 0;
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(a[i]) || !Number.isFinite(b[i]))
      throw Error("Nonfinite output");
    const d = a[i] - b[i];
    s += d * d;
    r += b[i] * b[i];
    max = Math.max(max, Math.abs(d));
  }
  return { relRMS: Math.sqrt(s / Math.max(r, 1e-30)), maxAbs: max };
}
export async function fixture(read, precision, L) {
  const meta = JSON.parse(
      new TextDecoder().decode(
        await read(`fixtures/oracle_${precision}_L${L}.json`),
      ),
    ),
    v = {};
  for (const [name, t] of Object.entries(meta.tensors)) {
    const bytes = await read(t.file);
    if ((await sha256(bytes)) !== t.sha256)
      throw Error("Oracle SHA-256 failure: " + name);
    v[name] = new Float32Array(bytes);
    if (v[name].length !== t.shape.reduce((a, b) => a * b, 1))
      throw Error("Oracle shape: " + name);
  }
  return { meta, v };
}
export function check(name, a, b, limit = 1e-4) {
  const m = metrics(a, b);
  if (!(m.relRMS < limit))
    throw Error(`${name}: relRMS ${m.relRMS}, maxAbs ${m.maxAbs}`);
  return { name, ...m };
}
export function firstAtoms(a) {
  const L = a.length / (37 * 3),
    r = new Float32Array(L * 15);
  for (let i = 0; i < L; i++) r.set(a.subarray(i * 111, i * 111 + 15), i * 15);
  return r;
}
