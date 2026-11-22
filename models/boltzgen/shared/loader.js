export const sizeOf = (shape) => shape.reduce((a, b) => a * b, 1);
export function halfToFloat(h) {
  const s = h & 32768 ? -1 : 1,
    e = (h >>> 10) & 31,
    m = h & 1023;
  return (
    s *
    (e === 0
      ? m * 2 ** -24
      : e === 31
        ? m
          ? NaN
          : Infinity
        : (1 + m / 1024) * 2 ** (e - 15))
  );
}
export function parseSafetensors(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 10)
    throw Error("Invalid safetensors");
  const view = new DataView(buffer),
    length = Number(view.getBigUint64(0, true));
  if (
    !Number.isSafeInteger(length) ||
    length < 2 ||
    length + 8 > buffer.byteLength
  )
    throw Error("Invalid header length");
  const header = JSON.parse(
      new TextDecoder().decode(new Uint8Array(buffer, 8, length)),
    ),
    out = {};
  const ranges = [];
  for (const [name, record] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const { shape, dtype, data_offsets: offsets } = record;
    const width = {
      F32: 4,
      F16: 2,
      I8: 1,
      U8: 1,
      BOOL: 1,
      I32: 4,
      I64: 8,
      F64: 8,
    }[dtype];
    if (
      !width ||
      !Array.isArray(shape) ||
      shape.some((d) => !Number.isSafeInteger(d) || d < 0) ||
      !Array.isArray(offsets)
    )
      throw Error(`Invalid tensor ${name}`);
    const [a, b] = offsets;
    if (
      !Number.isSafeInteger(a) ||
      !Number.isSafeInteger(b) ||
      a < 0 ||
      b < a ||
      b + 8 + length > buffer.byteLength ||
      b - a !== sizeOf(shape) * width
    )
      throw Error(`Invalid offsets ${name}`);
    ranges.push([a, b]);
    out[name] = {
      shape,
      dtype,
      bytes: new Uint8Array(buffer, 8 + length + a, b - a),
    };
  }
  ranges.sort((a, b) => a[0] - b[0]);
  for (let i = 1; i < ranges.length; i++)
    if (ranges[i][0] < ranges[i - 1][1])
      throw Error("Overlapping tensor offsets");
  return out;
}
export function decodeTensor(t, scale) {
  const v = new DataView(
      t.bytes.buffer,
      t.bytes.byteOffset,
      t.bytes.byteLength,
    ),
    data = new Float32Array(sizeOf(t.shape));
  for (let i = 0; i < data.length; i++) {
    switch (t.dtype) {
      case "F32":
        data[i] = v.getFloat32(i * 4, true);
        break;
      case "F64":
        data[i] = v.getFloat64(i * 8, true);
        break;
      case "F16":
        data[i] = halfToFloat(v.getUint16(i * 2, true));
        break;
      case "I64":
        data[i] = Number(v.getBigInt64(i * 8, true));
        break;
      case "I32":
        data[i] = v.getInt32(i * 4, true);
        break;
      case "I8":
        data[i] =
          v.getInt8(i) * (scale?.[Math.floor(i / (t.shape[1] ?? 1))] ?? 1);
        break;
      default:
        data[i] = v.getUint8(i);
    }
  }
  return data;
}
export async function sha256(bytes) {
  return Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
    (x) => x.toString(16).padStart(2, "0"),
  ).join("");
}
export function createReader({ read, cache = "none", signal } = {}) {
  if (!["none", "cache", "opfs"].includes(cache))
    throw Error(`Unknown cache ${cache}`);
  return async function (url, expected) {
    signal?.throwIfAborted();
    let store, key, bytes;
    if (read) bytes = await read(url);
    else {
      try {
        if (cache === "cache") {
          store = await caches.open("boltzgen-v1");
          bytes = await (await store.match(url))?.arrayBuffer();
        }
        if (cache === "opfs") {
          store = await (
            await navigator.storage.getDirectory()
          ).getDirectoryHandle("boltzgen-v1", { create: true });
          key = await sha256(new TextEncoder().encode(url));
          bytes = await (
            await (await store.getFileHandle(key)).getFile()
          ).arrayBuffer();
        }
      } catch {}
      if (!bytes) {
        const response = await fetch(url, { signal });
        if (!response.ok) throw Error(`HTTP ${response.status}: ${url}`);
        bytes = await response.arrayBuffer();
        if (
          expected &&
          (bytes.byteLength !== expected.byteLength ||
            (await sha256(bytes)) !== expected.sha256)
        )
          throw Error(`Integrity failure: ${url}`);
        try {
          if (cache === "cache" && store)
            await store.put(url, new Response(bytes));
          if (cache === "opfs" && store) {
            const writer = await (
              await store.getFileHandle(key, { create: true })
            ).createWritable();
            await writer.write(bytes);
            await writer.close();
          }
        } catch {}
      }
    }
    signal?.throwIfAborted();
    if (
      expected &&
      (bytes.byteLength !== expected.byteLength ||
        (await sha256(bytes)) !== expected.sha256)
    )
      throw Error(`Integrity failure: ${url}`);
    return bytes;
  };
}
export async function loadBundle(
  baseUrl,
  { onTensor, onProgress, ...options } = {},
) {
  const read = createReader(options),
    url = baseUrl.replace(/\/$/, "");
  const manifest = JSON.parse(
    new TextDecoder().decode(await read(url + "/manifest.json")),
  );
  if (manifest.format !== "boltzgen-safetensors-v1")
    throw Error("Unknown model bundle");
  let loaded = 0;
  for (const shard of manifest.shards) {
    options.signal?.throwIfAborted();
    if (shard.byteLength >= 25_000_000) throw Error("Oversized shard");
    const tensors = parseSafetensors(await read(url + "/" + shard.file, shard));
    for (const [name, t] of Object.entries(tensors)) {
      if (name.endsWith(".__scale")) continue;
      const scale = tensors[name + ".__scale"];
      await onTensor(name, t, scale);
    }
    loaded += shard.byteLength;
    onProgress?.({ loaded, total: manifest.total_bytes });
  }
  return manifest;
}
export function diffStats(actual, expected) {
  if (actual.length !== expected.length) throw Error("Shape mismatch");
  let err = 0,
    ref = 0,
    maxAbs = 0;
  for (let i = 0; i < actual.length; i++) {
    if (!Number.isFinite(actual[i]) || !Number.isFinite(expected[i]))
      throw Error(`Nonfinite output ${i}`);
    const d = actual[i] - expected[i];
    err += d * d;
    ref += expected[i] * expected[i];
    maxAbs = Math.max(maxAbs, Math.abs(d));
  }
  return { relRMS: Math.sqrt(err / Math.max(ref, 1e-30)), maxAbs };
}
