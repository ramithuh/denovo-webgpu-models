import { fileURLToPath } from "node:url";
import { readFile } from "node:fs/promises";
export const root = fileURLToPath(new URL("../", import.meta.url));
export async function localRead(url) {
  const b = await readFile(url);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}
export function metric(actual, expected) {
  if (!actual || actual.length !== expected.length)
    throw Error("Trace shape mismatch");
  let error = 0,
    norm = 0,
    maxAbs = 0;
  for (let i = 0; i < actual.length; i++) {
    const d = actual[i] - expected[i];
    error += d * d;
    norm += expected[i] * expected[i];
    maxAbs = Math.max(maxAbs, Math.abs(d));
  }
  return { relRMS: Math.sqrt(error / Math.max(norm, 1e-30)), maxAbs };
}
export async function fixture(name) {
  const meta = JSON.parse(
      await readFile(root + `fixtures/oracle_${name}.json`),
    ),
    bin = await localRead(root + "fixtures/" + meta.binary);
  const arrays = (entries) =>
    Object.fromEntries(
      Object.entries(entries).map(([k, v]) => [
        k,
        new Float32Array(bin.slice(v.offset, v.offset + v.length * 4)),
      ]),
    );
  return {
    meta,
    input: { length: meta.length, ...arrays(meta.inputs) },
    expected: arrays(meta.traces),
  };
}
