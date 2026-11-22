import { readFile } from "node:fs/promises";
export const root = new URL("../", import.meta.url);
export const read = async (url) => {
  const b = await readFile(url instanceof URL ? url : new URL(url, root));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
export async function fixture(precision, index = 0) {
  const j = JSON.parse(
      await readFile(
        new URL(`fixtures/oracle_${precision}.json`, root),
        "utf8",
      ),
    ),
    c = j.cases[index],
    b = await read(new URL(c.file, root));
  const hash = Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", b)),
    (x) => x.toString(16).padStart(2, "0"),
  ).join("");
  if (hash !== c.sha256) throw Error("Fixture checksum mismatch");
  const get = (k) => {
    const t = c.tensors[k];
    if (!t) throw Error(`Missing tensor ${k}`);
    return new Float32Array(b, t.offset, t.byteLength / 4).slice();
  };
  const features = Object.fromEntries(
    Object.keys(c.tensors)
      .filter((k) => k.startsWith("features."))
      .map((k) => [k.slice(9), get(k)]),
  );
  return { c, get, features };
}
