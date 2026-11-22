import { readFile } from "node:fs/promises";
export const root = new URL("../", import.meta.url);
export async function read(url) {
  const b = await readFile(url instanceof URL ? url : new URL(url, root));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}
export const readJSON = async (path) =>
  JSON.parse(new TextDecoder().decode(await read(path)));
export async function oracle(c, precision) {
  const stem = `fixtures/oracle_${c.same_as ?? c.id}_${precision}`,
    meta = await readJSON(stem + ".json"),
    binary = await read(stem + ".bin"),
    traces = {};
  for (const [name, t] of Object.entries(meta.tensors))
    traces[name] = new (t.dtype === "I32" ? Int32Array : Float32Array)(
      binary,
      t.offset,
      t.byteLength / 4,
    );
  return { meta, traces };
}
export function relRMS(a, b) {
  if (a.length !== b.length)
    throw new Error(`Length mismatch ${a.length} / ${b.length}`);
  let err = 0,
    ref = 0,
    maxAbs = 0;
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(a[i])) throw new Error(`Nonfinite at ${i}`);
    err += (a[i] - b[i]) ** 2;
    ref += b[i] ** 2;
    maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
  }
  return { relRMS: Math.sqrt(err / Math.max(ref, 1e-30)), maxAbs };
}
