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
    throw new Error("Invalid safetensors buffer");
  const view = new DataView(buffer);
  const n = Number(view.getBigUint64(0, true));
  if (!Number.isSafeInteger(n) || n < 2 || n + 8 > buffer.byteLength)
    throw new Error("Invalid safetensors header");
  const header = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buffer, 8, n)),
  );
  const tensors = {};
  for (const [name, rec] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    if (
      !Array.isArray(rec.shape) ||
      !rec.shape.length ||
      rec.shape.some((d) => !Number.isSafeInteger(d) || d <= 0) ||
      !Array.isArray(rec.data_offsets) ||
      rec.data_offsets.length !== 2
    )
      throw new Error(`Invalid tensor: ${name}`);
    const [a, b] = rec.data_offsets,
      bytesPerElement = { F32: 4, F16: 2, I8: 1, I32: 4 }[rec.dtype];
    if (
      !bytesPerElement ||
      !Number.isSafeInteger(a) ||
      !Number.isSafeInteger(b) ||
      a < 0 ||
      b < a ||
      8 + n + b > buffer.byteLength ||
      b - a !== rec.shape.reduce((a, b) => a * b, 1) * bytesPerElement
    )
      throw new Error(`Invalid tensor: ${name}`);
    tensors[name] = {
      dtype: rec.dtype,
      shape: rec.shape,
      bytes: new Uint8Array(buffer, 8 + n + a, b - a),
    };
  }
  return tensors;
}
export function decodeTensor(t) {
  const count = t.shape.reduce((a, b) => a * b, 1),
    out = new Float32Array(count);
  const v = new DataView(
    t.bytes.buffer,
    t.bytes.byteOffset,
    t.bytes.byteLength,
  );
  const width = t.shape[1] ?? 1;
  for (let i = 0; i < count; i++)
    out[i] =
      t.dtype === "F32"
        ? v.getFloat32(i * 4, true)
        : t.dtype === "F16"
          ? halfToFloat(v.getUint16(i * 2, true))
          : v.getInt8(i) * t.scales[Math.floor(i / width)];
  return out;
}
export function diffStats(a, b) {
  if (a.length !== b.length) throw new Error("Shape mismatch");
  let d = 0,
    r = 0,
    maxAbs = 0,
    sa = 0,
    sb = 0,
    aa = 0,
    bb = 0,
    ab = 0;
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(a[i]) || !Number.isFinite(b[i]))
      throw new Error(`Nonfinite value at ${i}`);
    const e = a[i] - b[i];
    d += e * e;
    r += b[i] * b[i];
    maxAbs = Math.max(maxAbs, Math.abs(e));
    sa += a[i];
    sb += b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
    ab += a[i] * b[i];
  }
  return {
    relRms: Math.sqrt(d / Math.max(r, 1e-30)),
    maxAbs,
    correlation:
      (ab - (sa * sb) / a.length) /
      Math.sqrt(
        Math.max(
          1e-30,
          (aa - (sa * sa) / a.length) * (bb - (sb * sb) / a.length),
        ),
      ),
  };
}
export async function loadWeights(
  baseUrl,
  { read, signal, onProgress, cache = "none", onTensor } = {},
) {
  if (!["none", "cache", "opfs"].includes(cache))
    throw new RangeError("Unknown cache mode");
  const fetchBytes = async (url) => {
    signal?.throwIfAborted();
    if (read) return read(url);
    let storage, key, cached;
    try {
      if (cache === "cache" && globalThis.caches) {
        storage = await caches.open("genie2-v1");
        cached = await storage.match(url);
      }
      if (cache === "opfs" && globalThis.navigator?.storage?.getDirectory) {
        storage = await (
          await navigator.storage.getDirectory()
        ).getDirectoryHandle("genie2-v1", { create: true });
        key = Array.from(
          new Uint8Array(
            await crypto.subtle.digest(
              "SHA-256",
              new TextEncoder().encode(url),
            ),
          ),
          (x) => x.toString(16).padStart(2, "0"),
        ).join("");
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
      key = null;
      onProgress?.({ phase: "cache-warning", message: e.message });
    }
    if (cached) return cached.arrayBuffer();
    const response = await fetch(url, { signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${url}`);
    const data = await response.arrayBuffer();
    if (storage) {
      try {
        if (key) {
          const writer = await (
            await storage.getFileHandle(key, { create: true })
          ).createWritable();
          await writer.write(data);
          await writer.close();
        } else await storage.put(url, new Response(data));
      } catch (e) {
        onProgress?.({ phase: "cache-warning", message: e.message });
      }
    }
    return data;
  };
  const base = baseUrl.replace(/\/$/, "");
  const manifest = JSON.parse(
    new TextDecoder().decode(await fetchBytes(`${base}/manifest.json`)),
  );
  validateManifest(manifest);
  const byShard = new Map(),
    pending = new Map(),
    weights = {};
  for (const [name, rec] of Object.entries(manifest.tensors)) {
    for (const part of rec.parts ?? [rec]) {
      if (!byShard.has(part.shard)) byShard.set(part.shard, []);
      byShard.get(part.shard).push({ name, rec, part });
    }
  }
  let loaded = 0,
    completed = 0;
  for (const shard of manifest.shards) {
    signal?.throwIfAborted();
    const data = await fetchBytes(`${base}/${shard.file}`);
    signal?.throwIfAborted();
    if (data.byteLength !== shard.byteLength || shard.byteLength > 25_000_000)
      throw new Error(`Shard size mismatch: ${shard.file}`);
    if (!globalThis.crypto?.subtle)
      throw new Error("Web Crypto SHA-256 is required");
    {
      const hash = Array.from(
        new Uint8Array(await crypto.subtle.digest("SHA-256", data)),
        (x) => x.toString(16).padStart(2, "0"),
      ).join("");
      if (hash !== shard.sha256)
        throw new Error(`Shard checksum mismatch: ${shard.file}`);
    }
    const parsed = parseSafetensors(data);
    for (const { name, rec, part } of byShard.get(shard.file) ?? []) {
      const raw = parsed[part.key ?? name];
      if (
        !raw ||
        raw.dtype !== part.dtype ||
        raw.shape.toString() !== part.shape.toString()
      )
        throw new Error(`Manifest mismatch: ${name}`);
      let t = pending.get(name);
      if (!t) {
        t = {
          dtype: rec.dtype,
          shape: rec.shape,
          bytes: new Uint8Array(
            rec.shape.reduce((a, b) => a * b, 1) *
              { F32: 4, F16: 2, I8: 1 }[rec.dtype],
          ),
          remaining: (rec.parts ?? [rec]).length,
        };
        if (rec.dtype === "I8") t.scales = new Float32Array(rec.shape[0]);
        pending.set(name, t);
      }
      const row = part.row_start ?? 0,
        stride = (rec.shape[1] ?? 1) * { F32: 4, F16: 2, I8: 1 }[rec.dtype];
      t.bytes.set(raw.bytes, row * stride);
      if (part.quant) {
        const scale = parsed[part.quant.scale_tensor];
        if (
          !scale ||
          scale.dtype !== "F32" ||
          scale.shape.length !== 1 ||
          scale.shape[0] !== part.shape[0]
        )
          throw new Error(`Invalid colocated scale: ${name}`);
        const values = decodeTensor(scale);
        if (values.some((v) => !Number.isFinite(v) || v <= 0))
          throw new Error(`Invalid scale value: ${name}`);
        t.scales.set(values, row);
      }
      if (--t.remaining === 0) {
        completed++;
        delete t.remaining;
        pending.delete(name);
        if (onTensor) await onTensor(name, t);
        else weights[name] = t;
      }
    }
    loaded += shard.bytes;
    onProgress?.({
      phase: "weights",
      loaded,
      total: manifest.total_bytes,
      file: shard.file,
    });
  }
  signal?.throwIfAborted();
  if (pending.size || completed !== manifest.num_tensors)
    throw new Error("Incomplete bundle");
  return { manifest, config: manifest.config, weights };
}

export function validateManifest(m) {
  if (m.format !== "genie2-safetensors-v1")
    throw new Error("Unsupported Genie 2 bundle format");
  if (
    !m.tensors ||
    !Array.isArray(m.shards) ||
    !m.shards.length ||
    Object.keys(m.tensors).length !== m.num_tensors
  )
    throw new Error("Invalid tensor manifest");
  const shards = new Set();
  let bytes = 0,
    params = 0;
  for (const s of m.shards) {
    if (
      !/^[a-zA-Z0-9_.-]+\.safetensors$/.test(s.file) ||
      shards.has(s.file) ||
      !Number.isSafeInteger(s.byteLength) ||
      s.byteLength < 10 ||
      s.byteLength > 25_000_000 ||
      s.bytes !== s.byteLength ||
      !/^[0-9a-f]{64}$/.test(s.sha256)
    )
      throw new Error("Invalid shard manifest");
    shards.add(s.file);
    bytes += s.byteLength;
  }
  if (bytes !== m.total_bytes) throw new Error("Bundle byte count mismatch");
  for (const [name, r] of Object.entries(m.tensors)) {
    if (
      !["F32", "F16", "I8"].includes(r.dtype) ||
      !Array.isArray(r.shape) ||
      r.shape.length < 1 ||
      r.shape.length > 2 ||
      r.shape.some((d) => !Number.isSafeInteger(d) || d <= 0)
    )
      throw new Error(`Invalid tensor shape/dtype: ${name}`);
    params += r.shape.reduce((a, b) => a * b, 1);
    let row = 0;
    for (const p of r.parts ?? [r]) {
      if (
        !shards.has(p.shard) ||
        p.dtype !== r.dtype ||
        !Array.isArray(p.shape) ||
        p.shape.length !== r.shape.length ||
        p.shape.some((d) => !Number.isSafeInteger(d) || d <= 0) ||
        p.row_start !== row ||
        p.shape.slice(1).toString() !== r.shape.slice(1).toString()
      )
        throw new Error(`Invalid multipart rows: ${name}`);
      row += p.shape[0];
      if (
        r.dtype === "I8" &&
        (!p.quant ||
          p.quant.scheme !== "symmetric_per_row" ||
          p.quant.axis !== 0 ||
          typeof p.quant.scale_tensor !== "string" ||
          r.shape.length !== 2)
      )
        throw new Error(`Missing per-row quantization: ${name}`);
    }
    if (row !== r.shape[0]) throw new Error(`Incomplete tensor rows: ${name}`);
  }
  if (params !== m.total_params) throw new Error("Parameter count mismatch");
}
