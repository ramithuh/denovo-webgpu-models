import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  decodeSafetensors,
  assembleWeights,
} from "../shared/ligandmpnn/weights.js";
export const ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
);
export const meta = JSON.parse(
  fs.readFileSync(path.join(ROOT, "fixtures/oracle_ligandmpnn.json")),
);
const bin = fs.readFileSync(path.join(ROOT, "fixtures/oracle_ligandmpnn.bin"));
export const readOracle = (name) => {
  const t = meta.tensors[name];
  if (!t) throw Error("Missing oracle " + name);
  const b = bin.buffer.slice(
    bin.byteOffset + t.byteOffset,
    bin.byteOffset + t.byteOffset + t.byteLength,
  );
  return t.dtype === "I32" ? new Int32Array(b) : new Float32Array(b);
};
export function loadWeights(
  precision = "fp32",
  variant = "ligandmpnn_v_32_010_25",
) {
  const folder = path.join(ROOT, "weights", variant, precision),
    manifest = JSON.parse(fs.readFileSync(path.join(folder, "manifest.json"))),
    shards = {};
  for (const s of manifest.shards) {
    const raw = fs.readFileSync(path.join(folder, s.file));
    shards[s.file] = decodeSafetensors(
      raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength),
    );
  }
  return assembleWeights(manifest, shards);
}
export const structure = fs.readFileSync(
  path.join(ROOT, "fixtures/holo_1BC8.pdb"),
  "utf8",
);
export function stats(got, ref) {
  if (got.length !== ref.length)
    throw Error(`Length ${got.length} != ${ref.length}`);
  let e = 0,
    r = 0,
    max = 0;
  for (let i = 0; i < ref.length; i++) {
    const d = got[i] - ref[i];
    e += d * d;
    r += ref[i] * ref[i];
    max = Math.max(max, Math.abs(d));
  }
  return { relRMS: Math.sqrt(e / Math.max(r, 1e-30)), maxAbs: max };
}
