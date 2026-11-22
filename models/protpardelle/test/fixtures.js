export async function oracleBinary(
  meta,
  baseUrl,
  read = async (url) => {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`Missing oracle ${url}`);
    return r.arrayBuffer();
  },
) {
  const names = meta.binary_parts ?? [`oracle_${meta.variant}.bin`],
    parts = [];
  for (const name of names)
    parts.push(new Uint8Array(await read(baseUrl + name)));
  const result = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let offset = 0;
  for (const part of parts) {
    result.set(part, offset);
    offset += part.length;
  }
  return result.buffer;
}
