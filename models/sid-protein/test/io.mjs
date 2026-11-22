import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
export const root = new URL("../", import.meta.url);
export const read = async (u) => {
  const a = await readFile(u instanceof URL ? u : new URL(u, root));
  return a.buffer.slice(a.byteOffset, a.byteOffset + a.byteLength);
};
export const json = async (u) =>
  JSON.parse(new TextDecoder().decode(await read(u)));
export const floats = async (r) => {
  const a = await read(r.file);
  if (
    a.byteLength !== r.byteLength ||
    createHash("sha256").update(new Uint8Array(a)).digest("hex") !== r.sha256
  )
    throw new Error("Fixture checksum mismatch " + r.file);
  return new Float32Array(a);
};
