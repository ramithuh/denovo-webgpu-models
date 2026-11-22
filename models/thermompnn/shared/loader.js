export function checkAbort(signal) {
  if (signal?.aborted)
    throw signal.reason ?? new DOMException("Aborted", "AbortError");
}
export function halfToFloat(h) {
  const s = h >>> 15 ? -1 : 1,
    e = (h >>> 10) & 31,
    m = h & 1023;
  return e === 0
    ? (s * 2 ** -14 * m) / 1024
    : e === 31
      ? m
        ? NaN
        : s * Infinity
      : s * 2 ** (e - 15) * (1 + m / 1024);
}
const sizeOf = { F32: 4, F16: 2, I8: 1, I32: 4 };
export function parseSafetensors(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 8)
    throw new Error("Invalid safetensors buffer");
  const v = new DataView(buffer),
    n = Number(v.getBigUint64(0, true));
  if (!Number.isSafeInteger(n) || n < 2 || n > buffer.byteLength - 8)
    throw new Error("Invalid safetensors header length");
  const header = JSON.parse(
      new TextDecoder().decode(new Uint8Array(buffer, 8, n)),
    ),
    out = {};
  const ranges = [];
  for (const [key, t] of Object.entries(header)) {
    if (key === "__metadata__") continue;
    const [a, b] = t.data_offsets ?? [],
      count = t.shape?.reduce((a, b) => a * b, 1);
    if (
      !sizeOf[t.dtype] ||
      !t.shape.every((x) => Number.isSafeInteger(x) && x > 0) ||
      !Number.isSafeInteger(a) ||
      !Number.isSafeInteger(b) ||
      a < 0 ||
      b < a ||
      8 + n + b > buffer.byteLength ||
      b - a !== count * sizeOf[t.dtype]
    )
      throw new Error(`Invalid tensor: ${key}`);
    ranges.push([a, b]);
    out[key] = {
      shape: t.shape,
      dtype: t.dtype,
      bytes: new Uint8Array(buffer.slice(8 + n + a, 8 + n + b)),
    };
  }
  ranges.sort((a, b) => a[0] - b[0]);
  for (let i = 1; i < ranges.length; i++)
    if (ranges[i][0] < ranges[i - 1][1])
      throw new Error("Overlapping safetensors tensors");
  return out;
}
export function tensorToFloat(t) {
  const n = t.shape.reduce((a, b) => a * b, 1),
    a = new Float32Array(n),
    v = new DataView(t.bytes.buffer, t.bytes.byteOffset, t.bytes.byteLength);
  for (let i = 0; i < n; i++)
    a[i] =
      t.dtype === "F32"
        ? v.getFloat32(i * 4, true)
        : t.dtype === "F16"
          ? halfToFloat(v.getUint16(i * 2, true))
          : v.getInt8(i) * (t.scale?.[Math.floor(i / (n / t.shape[0]))] ?? 1);
  return a;
}
async function digest(buffer) {
  if (!globalThis.crypto?.subtle)
    throw new Error("Web Crypto is required for SHA-256 validation");
  return [...new Uint8Array(await crypto.subtle.digest("SHA-256", buffer))]
    .map((x) => x.toString(16).padStart(2, "0"))
    .join("");
}
async function storage(cache) {
  try {
    if (cache === "cache") {
      const c = await globalThis.caches.open("thermompnn-v1");
      return {
        get: async (u) => (await c.match(u))?.arrayBuffer(),
        put: async (u, b) => c.put(u, new Response(b)),
      };
    }
    if (cache === "opfs") {
      const root = await navigator.storage.getDirectory(),
        dir = await root.getDirectoryHandle("thermompnn-v1", { create: true });
      const key = async (u) => digest(new TextEncoder().encode(u).buffer);
      return {
        get: async (u) =>
          (
            await (await dir.getFileHandle(await key(u))).getFile()
          ).arrayBuffer(),
        put: async (u, b) => {
          const f = await dir.getFileHandle(await key(u), { create: true }),
            w = await f.createWritable();
          try {
            await w.write(b);
            await w.close();
          } catch (e) {
            await w.abort().catch(() => {});
            throw e;
          }
        },
      };
    }
  } catch {
    /* Private Mode / quota: fetching remains available. */
  }
  return null;
}
export async function loadWeights({
  baseUrl,
  precision = "fp32",
  read,
  cache = "none",
  signal,
  onProgress,
} = {}) {
  if (!baseUrl) throw new Error("baseUrl is required");
  if (!["fp32", "fp16", "int8"].includes(precision))
    throw new Error("Invalid precision");
  if (!["none", "cache", "opfs"].includes(cache))
    throw new Error("Invalid cache");
  const root = String(baseUrl).replace(/\/$/, ""),
    base = /(fp32|fp16|int8)$/.test(root) ? root : `${root}/${precision}`;
  const st = await storage(cache);
  async function get(url, expected) {
    checkAbort(signal);
    async function valid(b) {
      if (!(b instanceof ArrayBuffer))
        throw new Error("read(url) must return ArrayBuffer");
      if (!expected) {
        const m = JSON.parse(new TextDecoder().decode(b));
        if (
          m.format !== "thermompnn-safetensors-v1" ||
          m.model !== "ThermoMPNN" ||
          m.precision !== precision ||
          !m.shards?.length ||
          !m.tensors
        )
          throw new Error("Invalid ThermoMPNN manifest");
      }
      if (
        expected &&
        (b.byteLength !== expected.byteLength ||
          (await digest(b)) !== expected.sha256)
      )
        throw new Error(`Shard integrity failure: ${url}`);
      return b;
    }
    if (st) {
      let hit;
      try {
        hit = await st.get(url);
      } catch {}
      if (hit) {
        try {
          return await valid(hit);
        } catch {
          /* corrupt cache: refetch */
        }
      }
    }
    let b;
    if (read) b = await read(url);
    else {
      const response = await fetch(url, { signal });
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${url}`);
      b = await response.arrayBuffer();
    }
    checkAbort(signal);
    await valid(b);
    if (st)
      try {
        await st.put(url, b);
      } catch {}
    return b;
  }
  const manifest = JSON.parse(
    new TextDecoder().decode(await get(`${base}/manifest.json`)),
  );
  if (
    manifest.format !== "thermompnn-safetensors-v1" ||
    manifest.model !== "ThermoMPNN" ||
    manifest.precision !== precision ||
    !manifest.shards?.length ||
    !manifest.tensors
  )
    throw new Error("Invalid ThermoMPNN manifest");
  const raw = new Map();
  let loaded = 0;
  for (const shard of manifest.shards) {
    if (
      !/^[\w.-]+\.safetensors$/.test(shard.file) ||
      shard.byteLength > 25000000 ||
      !Number.isSafeInteger(shard.byteLength) ||
      shard.byteLength <= 0 ||
      !/^[a-f0-9]{64}$/.test(shard.sha256)
    )
      throw new Error("Invalid shard metadata");
    const b = await get(`${base}/${shard.file}`, shard);
    raw.set(shard.file, parseSafetensors(b));
    loaded += b.byteLength;
    onProgress?.({ phase: "weights", loaded, total: manifest.total_bytes });
    checkAbort(signal);
  }
  if (loaded !== manifest.total_bytes)
    throw new Error("Manifest byte total mismatch");
  const tensors = {};
  let params = 0;
  for (const [name, entry] of Object.entries(manifest.tensors)) {
    const parts = entry.parts ?? [entry],
      shape = entry.shape,
      dtype = entry.dtype;
    if (
      !sizeOf[dtype] ||
      !Array.isArray(shape) ||
      !shape.every((x) => Number.isSafeInteger(x) && x > 0)
    )
      throw new Error(`Invalid tensor shape ${name}`);
    const count = shape.reduce((a, b) => a * b, 1),
      bytes = new Uint8Array(count * sizeOf[dtype]);
    const scale = dtype === "I8" ? new Float32Array(shape[0]) : undefined;
    let row = 0,
      offset = 0;
    for (const part of parts) {
      const t = raw.get(part.shard)?.[part.key];
      if (
        !t ||
        part.dtype !== dtype ||
        t.dtype !== dtype ||
        part.row_start !== row ||
        JSON.stringify(t.shape) !== JSON.stringify(part.shape) ||
        JSON.stringify(t.shape.slice(1)) !== JSON.stringify(shape.slice(1))
      )
        throw new Error(`Invalid multipart tensor ${name}`);
      if (offset + t.bytes.byteLength > bytes.byteLength)
        throw new Error(`Oversized tensor ${name}`);
      bytes.set(t.bytes, offset);
      offset += t.bytes.byteLength;
      if (scale) {
        const s = raw.get(part.shard)?.[part.quant?.scale_tensor];
        if (
          !s ||
          s.dtype !== "F32" ||
          s.shape.length !== 1 ||
          s.shape[0] !== t.shape[0] ||
          part.quant.scheme !== "symmetric_per_row"
        )
          throw new Error(`Missing colocated scales ${name}`);
        scale.set(tensorToFloat(s), row);
      }
      row += t.shape[0];
    }
    if (offset !== bytes.byteLength || row !== shape[0])
      throw new Error(`Incomplete tensor ${name}`);
    tensors[name] = { shape, dtype, bytes, scale };
    params += count;
  }
  if (
    params !== manifest.total_params ||
    Object.keys(tensors).length !== manifest.num_tensors
  )
    throw new Error("Manifest tensor totals mismatch");
  return { manifest, tensors };
}
