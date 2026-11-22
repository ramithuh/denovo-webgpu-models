import { expectedShapes } from "./spec.js";
export async function sha256(bytes) {
  return Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)),
    (x) => x.toString(16).padStart(2, "0"),
  ).join("");
}
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
    throw new Error("Invalid safetensors");
  const n = Number(new DataView(buffer).getBigUint64(0, true));
  if (!Number.isSafeInteger(n) || n < 2 || n + 8 > buffer.byteLength)
    throw new Error("Invalid header");
  const header = JSON.parse(
      new TextDecoder().decode(new Uint8Array(buffer, 8, n)),
    ),
    out = {},
    ranges = [];
  for (const [name, r] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const bytes = { F32: 4, F16: 2, I8: 1 }[r.dtype],
      [a, b] = r.data_offsets ?? [];
    if (
      !bytes ||
      !Array.isArray(r.shape) ||
      !r.shape.length ||
      r.shape.some((x) => !Number.isSafeInteger(x) || x <= 0) ||
      !Number.isSafeInteger(a) ||
      !Number.isSafeInteger(b) ||
      a < 0 ||
      b < a ||
      b + 8 + n > buffer.byteLength ||
      b - a !== r.shape.reduce((a, b) => a * b, 1) * bytes
    )
      throw new Error(`Invalid tensor ${name}`);
    ranges.push([a, b]);
    out[name] = {
      dtype: r.dtype,
      shape: r.shape,
      bytes: new Uint8Array(buffer, 8 + n + a, b - a),
    };
  }
  ranges.sort((a, b) => a[0] - b[0]);
  let end = 0;
  for (const [a, b] of ranges) {
    if (a !== end) throw new Error("Overlapping or noncontiguous tensors");
    end = b;
  }
  if (end !== buffer.byteLength - 8 - n)
    throw new Error("Trailing safetensors bytes");
  return out;
}
export function decodeTensor(t) {
  const out = new Float32Array(t.shape.reduce((a, b) => a * b, 1)),
    view = new DataView(t.bytes.buffer, t.bytes.byteOffset, t.bytes.byteLength);
  const stride = out.length / t.shape[0];
  for (let i = 0; i < out.length; i++)
    out[i] =
      t.dtype === "F32"
        ? view.getFloat32(i * 4, true)
        : t.dtype === "F16"
          ? halfToFloat(view.getUint16(i * 2, true))
          : view.getInt8(i) * t.scales[Math.floor(i / stride)];
  return out;
}
// Storage failures (including Safari Private Mode SecurityError/QuotaExceededError) fall back to fetch.
export function createReader({
  read,
  cache = "cache",
  signal,
  onProgress,
} = {}) {
  if (!["none", "cache", "opfs"].includes(cache))
    throw new RangeError("Invalid cache mode");
  return async (url) => {
    signal?.throwIfAborted();
    if (read) {
      const data = await read(url, { signal });
      signal?.throwIfAborted();
      return data;
    }
    let storage, key, hit;
    try {
      if (cache === "cache" && globalThis.caches) {
        storage = await caches.open("proteina-v1");
        hit = await storage.match(url);
        if (hit) return hit.arrayBuffer();
      } else if (
        cache === "opfs" &&
        globalThis.navigator?.storage?.getDirectory
      ) {
        storage = await (
          await navigator.storage.getDirectory()
        ).getDirectoryHandle("proteina-v1", { create: true });
        key = await sha256(new TextEncoder().encode(url));
        try {
          return await (
            await (await storage.getFileHandle(key)).getFile()
          ).arrayBuffer();
        } catch (e) {
          if (e.name !== "NotFoundError") throw e;
        }
      }
    } catch (e) {
      storage = null;
      onProgress?.({ phase: "cache-warning", message: e.message });
    }
    const response = await fetch(url, { signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${url}`);
    const data = await response.arrayBuffer();
    if (storage)
      try {
        if (key) {
          const writer = await (
            await storage.getFileHandle(key, { create: true })
          ).createWritable();
          try {
            await writer.write(data);
            await writer.close();
          } catch (e) {
            await writer.abort().catch(() => {});
            throw e;
          }
        } else await storage.put(url, new Response(data));
      } catch (e) {
        onProgress?.({ phase: "cache-warning", message: e.message });
      }
    signal?.throwIfAborted();
    return data;
  };
}
export function validateManifest(m) {
  if (
    m.format !== "proteina-safetensors-v1" ||
    m.model !== "proteina_v1.2_DFS_200M_notri" ||
    m.commit !== "a44b407daf6a5358e43cd68907f3e3f1cbc65fdc" ||
    !["fp32", "fp16", "int8"].includes(m.precision)
  )
    throw Error("Unsupported Proteina bundle");
  if (
    m.num_tensors !== 490 ||
    m.total_params !== 191102976 ||
    Object.keys(m.tensors ?? {}).length !== 490
  )
    throw Error("Incomplete Proteina bundle");
  if (
    m.config.token_dim !== 768 ||
    m.config.nlayers !== 15 ||
    m.config.update_pair_repr ||
    m.config.num_registers !== 10
  )
    throw Error("Unsupported architecture");
  const expected = expectedShapes();
  for (const [name, shape] of Object.entries(expected)) {
    const spec = m.tensors[name],
      compressed =
        shape.length === 2 &&
        name.endsWith(".weight") &&
        !name.includes("embedding_");
    const dtype = compressed
      ? { fp32: "F32", fp16: "F16", int8: "I8" }[m.precision]
      : "F32";
    if (
      !spec ||
      spec.shape?.toString() !== shape.toString() ||
      spec.dtype !== dtype
    )
      throw Error("Unsupported tensor " + name);
  }
  for (const [key, size] of [
    ["timeFreq", 128],
    ["indexDivisor", 64],
    ["scBins", 127],
    ["xtBins", 63],
  ])
    if (
      m.constants?.[key]?.length !== size ||
      m.constants[key].some((x) => !Number.isFinite(x))
    )
      throw Error("Invalid feature constants");
  const files = new Set();
  let size = 0,
    count = 0;
  for (const s of m.shards ?? []) {
    if (
      !/^[\w.-]+\.safetensors$/.test(s.file) ||
      files.has(s.file) ||
      !Number.isSafeInteger(s.byteLength) ||
      s.byteLength < 10 ||
      s.byteLength > 25000000 ||
      !/^[a-f0-9]{64}$/.test(s.sha256)
    )
      throw Error("Invalid shard");
    files.add(s.file);
    size += s.byteLength;
  }
  for (const [name, t] of Object.entries(m.tensors)) {
    if (
      !["F32", "F16", "I8"].includes(t.dtype) ||
      !Array.isArray(t.shape) ||
      t.shape.length < 1 ||
      t.shape.length > 2 ||
      t.shape.some((n) => !Number.isSafeInteger(n) || n < 1)
    )
      throw Error(`Invalid tensor ${name}`);
    let row = 0;
    for (const p of t.parts ?? []) {
      if (
        !files.has(p.shard) ||
        typeof p.key !== "string" ||
        p.rowStart !== row ||
        !Number.isSafeInteger(p.rowEnd) ||
        p.rowEnd <= row ||
        p.rowEnd > t.shape[0] ||
        (t.dtype === "I8" && p.scale !== p.key + ".__scale")
      )
        throw Error("Invalid row split");
      row = p.rowEnd;
    }
    if (row !== t.shape[0]) throw Error("Missing tensor rows");
    count += t.shape.reduce((a, b) => a * b, 1);
  }
  if (size !== m.total_bytes || count !== m.total_params)
    throw Error("Manifest totals mismatch");
}
export async function loadWeights(baseUrl, options = {}) {
  if (typeof baseUrl !== "string") throw TypeError("baseUrl is required");
  const read = createReader(options),
    base = baseUrl.replace(/\/$/, ""),
    manifest = JSON.parse(
      new TextDecoder().decode(await read(base + "/manifest.json")),
    );
  validateManifest(manifest);
  if (
    options.expectedPrecision &&
    manifest.precision !== options.expectedPrecision
  )
    throw Error("Precision mismatch");
  const weights = {},
    partial = new Map();
  let completed = 0,
    loaded = 0;
  for (const shard of manifest.shards) {
    options.signal?.throwIfAborted();
    const data = await read(base + "/" + shard.file);
    if (
      data.byteLength !== shard.byteLength ||
      (await sha256(data)) !== shard.sha256
    )
      throw Error(`Checksum mismatch: ${shard.file}`);
    const parsed = parseSafetensors(data);
    for (const [name, spec] of Object.entries(manifest.tensors))
      for (let i = 0; i < spec.parts.length; i++) {
        const p = spec.parts[i];
        if (p.shard !== shard.file) continue;
        const t = parsed[p.key],
          shape = [p.rowEnd - p.rowStart, ...spec.shape.slice(1)];
        if (
          !t ||
          t.dtype !== spec.dtype ||
          t.shape.toString() !== shape.toString()
        )
          throw Error(`Tensor mismatch: ${name}`);
        if (t.dtype === "I8") {
          const s = parsed[p.scale];
          if (
            !s ||
            s.dtype !== "F32" ||
            s.shape.toString() !== String(shape[0])
          )
            throw Error("Missing colocated scale");
          t.scales = decodeTensor(s);
          if (t.scales.some((v) => !Number.isFinite(v) || v <= 0))
            throw Error("Invalid scales");
        }
        let parts = partial.get(name);
        if (!parts) {
          parts = [];
          partial.set(name, parts);
        }
        parts[i] = t;
        if (parts.filter(Boolean).length === spec.parts.length) {
          let tensor = t;
          if (parts.length > 1) {
            const bytes = new Uint8Array(
              parts.reduce((n, t) => n + t.bytes.byteLength, 0),
            );
            let pos = 0;
            for (const x of parts) {
              bytes.set(x.bytes, pos);
              pos += x.bytes.byteLength;
            }
            tensor = { bytes, shape: spec.shape, dtype: spec.dtype };
            if (t.dtype === "I8") {
              tensor.scales = new Float32Array(spec.shape[0]);
              for (let j = 0; j < parts.length; j++)
                tensor.scales.set(parts[j].scales, spec.parts[j].rowStart);
            }
          }
          if (options.onTensor) await options.onTensor(name, tensor);
          else weights[name] = tensor;
          partial.delete(name);
          completed++;
        }
      }
    loaded += data.byteLength;
    options.onProgress?.({
      phase: "weights",
      loaded,
      total: manifest.total_bytes,
    });
  }
  options.signal?.throwIfAborted();
  if (completed !== 490) throw Error("Incomplete weights");
  return { manifest, weights };
}
