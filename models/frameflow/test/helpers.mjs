import { fileURLToPath } from "node:url";
import { readFile } from "node:fs/promises";
export const root = fileURLToPath(new URL("../", import.meta.url));
export const read = async (url) => {
  const b = await readFile(url);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
