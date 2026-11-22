import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
export const root = fileURLToPath(new URL("../", import.meta.url));
export async function read(url) {
  const b = await readFile(url);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}
export function arrays(metadata, binary) {
  return Object.fromEntries(
    Object.entries(metadata).map(([key, value]) => [
      key,
      value === null
        ? null
        : new (value.dtype === "I32" ? Int32Array : Float32Array)(
            binary.slice(value.offset, value.offset + value.length * 4),
          ),
    ]),
  );
}
export function inputFrom(data) {
  return Object.fromEntries(
    Object.entries({
      coords: "noisy_coords",
      noiseLevel: "noise_level",
      seqMask: "seq_mask",
      residueIndex: "residue_index",
      chainIndex: "chain_index",
      hotspotMask: "hotspot_mask",
      selfCond: "struct_self_cond",
      cropCond: "struct_crop_cond",
      seqCropCond: "seq_crop_cond",
      neighborIndices: "neighbor_indices",
    }).map(([key, value]) => [key, data[value]]),
  );
}
export function metrics(actual, expected) {
  if (actual.length !== expected.length)
    throw new Error(`Length mismatch ${actual.length}/${expected.length}`);
  let error = 0,
    norm = 0,
    maxAbs = 0;
  for (let i = 0; i < actual.length; i++) {
    const delta = actual[i] - expected[i];
    error += delta * delta;
    norm += expected[i] * expected[i];
    maxAbs = Math.max(maxAbs, Math.abs(delta));
  }
  return { relRMS: Math.sqrt(error / Math.max(norm, 1e-30)), maxAbs };
}
