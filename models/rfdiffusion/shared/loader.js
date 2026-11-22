// Dependency-free safetensors reader. Raw storage precision is preserved for GPU upload.
const TYPES = {
  F32: [4, Float32Array],
  F16: [2, Uint16Array],
  I8: [1, Int8Array],
};
const decoder = new TextDecoder();
export function abort(signal) {
  if (signal?.aborted)
    throw signal.reason ?? new DOMException("Aborted", "AbortError");
}
export async function sha256(bytes) {
  if (!globalThis.crypto?.subtle)
    throw new Error("Web Crypto SHA-256 is required");
  return Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
    (x) => x.toString(16).padStart(2, "0"),
  ).join("");
}
export function parseSafetensors(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 8)
    throw new Error("Invalid safetensors buffer");
  const view = new DataView(buffer),
    headerLength = Number(view.getBigUint64(0, true));
  if (
    !Number.isSafeInteger(headerLength) ||
    headerLength > buffer.byteLength - 8
  )
    throw new Error("Invalid safetensors header length");
  const header = JSON.parse(
      decoder.decode(new Uint8Array(buffer, 8, headerLength)),
    ),
    result = {};
  for (const [key, info] of Object.entries(header)) {
    if (key === "__metadata__") continue;
    const type = TYPES[info.dtype];
    if (
      !type ||
      !Array.isArray(info.shape) ||
      info.shape.some((d) => !Number.isInteger(d) || d < 0)
    )
      throw new Error(`Invalid tensor ${key}`);
    const [start, end] = info.data_offsets,
      count = info.shape.reduce((a, b) => a * b, 1);
    if (
      !Number.isSafeInteger(start) ||
      !Number.isSafeInteger(end) ||
      start < 0 ||
      end - start !== count * type[0] ||
      end > buffer.byteLength - 8 - headerLength
    )
      throw new Error(`Invalid tensor offsets: ${key}`);
    result[key] = {
      dtype: info.dtype,
      shape: info.shape,
      data: new type[1](
        buffer.slice(8 + headerLength + start, 8 + headerLength + end),
      ),
    };
  }
  return result;
}
export function half(value) {
  const sign = value & 32768 ? -1 : 1,
    exp = (value >> 10) & 31,
    mant = value & 1023;
  return (
    sign *
    (exp === 0
      ? mant * 2 ** -24
      : exp === 31
        ? mant
          ? NaN
          : Infinity
        : (1 + mant / 1024) * 2 ** (exp - 15))
  );
}
export function toFloat(tensor) {
  if (tensor.dtype === "F32") return tensor.data;
  const out = new Float32Array(tensor.data.length),
    width = tensor.shape[1];
  for (let i = 0; i < out.length; i++)
    out[i] =
      tensor.dtype === "F16"
        ? half(tensor.data[i])
        : tensor.data[i] * tensor.scale[Math.floor(i / width)];
  return out;
}
async function cachedRead(url, { cache, read, signal }) {
  abort(signal);
  let storage, key;
  try {
    if (cache === "cache" && globalThis.caches) {
      storage = await caches.open("rfdiffusion-v1");
      const hit = await storage.match(url);
      if (hit) return await hit.arrayBuffer();
    } else if (
      cache === "opfs" &&
      globalThis.navigator?.storage?.getDirectory
    ) {
      storage = await (
        await navigator.storage.getDirectory()
      ).getDirectoryHandle("rfdiffusion-v1", { create: true });
      key = await sha256(new TextEncoder().encode(url));
      try {
        return await (
          await (await storage.getFileHandle(key)).getFile()
        ).arrayBuffer();
      } catch (error) {
        if (error.name !== "NotFoundError") throw error;
      }
    }
  } catch {
    storage = undefined;
  }
  let data;
  if (read) data = await read(url);
  else {
    const response = await fetch(url, { signal });
    if (!response.ok)
      throw new Error(`Download failed (${response.status}): ${url}`);
    data = await response.arrayBuffer();
  }
  abort(signal);
  try {
    if (storage && cache === "cache")
      await storage.put(url, new Response(data.slice(0)));
    if (storage && cache === "opfs") {
      const writer = await (
        await storage.getFileHandle(key, { create: true })
      ).createWritable();
      try {
        await writer.write(data);
      } finally {
        await writer.close();
      }
    }
  } catch {
    /* Storage-denied contexts still execute with fetched data. */
  }
  return data;
}
export async function loadBundle({
  baseUrl,
  read,
  cache = "none",
  signal,
  onProgress,
} = {}) {
  if (!baseUrl) throw new Error("baseUrl is required");
  const root = baseUrl.replace(/\/$/, ""),
    options = { read, cache, signal };
  const manifest = JSON.parse(
    decoder.decode(await cachedRead(`${root}/manifest.json`, options)),
  );
  if (manifest.format !== "rfdiffusion-safetensors-v1")
    throw new Error("Unsupported model manifest");
  const shards = {},
    weights = {};
  let loaded = 0;
  for (const item of manifest.shards) {
    if (
      !/^[\w.-]+\.safetensors$/.test(item.file) ||
      item.byteLength > 25_000_000
    )
      throw new Error("Invalid shard record");
    const bytes = await cachedRead(`${root}/${item.file}`, options);
    if (
      bytes.byteLength !== item.byteLength ||
      (await sha256(bytes)) !== item.sha256
    )
      throw new Error(`Shard integrity failure: ${item.file}`);
    shards[item.file] = parseSafetensors(bytes);
    loaded += bytes.byteLength;
    onProgress?.({ phase: "weights", loaded, total: manifest.total_bytes });
  }
  for (const [name, entry] of Object.entries(manifest.tensors)) {
    const parts = entry.parts ?? [entry],
      [bytes, Type] = TYPES[entry.dtype] ?? [];
    if (!Type) throw new Error(`Unsupported storage for ${name}`);
    const count = entry.shape.reduce((a, b) => a * b, 1),
      data = new Type(count);
    const scales =
      entry.dtype === "I8" ? new Float32Array(entry.shape[0]) : undefined;
    let offset = 0,
      row = 0;
    for (const part of parts) {
      const raw = shards[part.shard]?.[part.key];
      if (
        !raw ||
        raw.dtype !== entry.dtype ||
        part.row_start !== row ||
        JSON.stringify(raw.shape) !== JSON.stringify(part.shape)
      )
        throw new Error(`Invalid part ${name}`);
      data.set(raw.data, offset);
      offset += raw.data.length;
      row += raw.shape[0] ?? 1;
      if (scales) {
        const scale = shards[part.shard][part.scale_tensor];
        if (
          !scale ||
          scale.dtype !== "F32" ||
          scale.data.length !== raw.shape[0]
        )
          throw new Error(`Missing colocated scales: ${name}`);
        scales.set(scale.data, part.row_start);
      }
    }
    if (offset !== count) throw new Error(`Incomplete tensor: ${name}`);
    weights[name] = {
      data,
      dtype: entry.dtype,
      shape: entry.shape,
      scale: scales,
      byteLength: count * bytes,
    };
  }
  abort(signal);
  return { manifest, weights };
}
