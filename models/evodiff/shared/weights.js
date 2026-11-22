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
export function validateManifest(m) {
  if (
    m.format !== "evodiff-safetensors-v1" ||
    m.model !== "OA_DM_38M" ||
    !["fp32", "fp16", "int8"].includes(m.precision)
  )
    throw new Error("Unsupported model bundle");
  if (
    !m.shards?.length ||
    Object.keys(m.tensors ?? {}).length !== 199 ||
    m.num_tensors !== 199 ||
    m.total_params !== 37890327
  )
    throw new Error("Incomplete model manifest");
  const files = new Set();
  let bytes = 0,
    count = 0;
  for (const s of m.shards) {
    if (
      !/^[\w.-]+\.safetensors$/.test(s.file) ||
      files.has(s.file) ||
      !Number.isSafeInteger(s.byteLength) ||
      s.byteLength < 10 ||
      s.byteLength > 25000000 ||
      !/^[a-f0-9]{64}$/.test(s.sha256)
    )
      throw new Error("Invalid shard manifest");
    files.add(s.file);
    bytes += s.byteLength;
  }
  for (const [name, r] of Object.entries(m.tensors)) {
    if (
      !files.has(r.shard) ||
      !["F32", "F16", "I8"].includes(r.dtype) ||
      !Array.isArray(r.shape) ||
      r.shape.length < 1 ||
      r.shape.length > 3 ||
      r.shape.some((x) => !Number.isSafeInteger(x) || x < 1)
    )
      throw new Error(`Invalid tensor manifest ${name}`);
    if (
      r.dtype === "I8" &&
      (r.quant?.scheme !== "symmetric_per_output_channel" ||
        r.quant.axis !== 0 ||
        r.quant.scale_tensor !== name + ".__scale" ||
        r.shape.length !== 3)
    )
      throw new Error("Invalid quantization");
    count += r.shape.reduce((a, b) => a * b, 1);
  }
  if (bytes !== m.total_bytes || count !== m.total_params)
    throw new Error("Invalid manifest totals");
  if (
    JSON.stringify(m.config?.dilations) !==
      JSON.stringify([
        1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128,
      ]) ||
    m.config.width !== 1024 ||
    m.config.activation !== "gelu_exact"
  )
    throw new Error("Unsupported architecture");
  const expected = {
    "embedder.embedder.weight": [31, 8],
    "embedder.up_embedder.conv.weight": [1024, 8, 1],
    "embedder.up_embedder.conv.bias": [1024],
    "decoder.conv.weight": [31, 1024, 1],
    "decoder.conv.bias": [31],
    "last_norm.weight": [1024],
    "last_norm.bias": [1024],
  };
  for (let i = 0; i < 16; i++) {
    const prefix = `embedder.layers.${i}.`;
    for (const [name, shape] of Object.entries({
      conv: [512, 512, 5],
      "sequence1.0": [1024],
      "sequence1.2.conv": [512, 1024, 1],
      "sequence1.3": [512],
      "sequence2.0": [512],
      "sequence2.2.conv": [1024, 512, 1],
    })) {
      expected[prefix + name + ".weight"] = shape;
      expected[prefix + name + ".bias"] = [shape[0]];
    }
  }
  for (const [name, shape] of Object.entries(expected)) {
    const r = m.tensors[name];
    const dtype =
      shape.length === 3
        ? { fp32: "F32", fp16: "F16", int8: "I8" }[m.precision]
        : "F32";
    if (!r || r.shape.toString() !== shape.toString() || r.dtype !== dtype)
      throw new Error(`Unexpected OA_DM_38M tensor: ${name}`);
  }
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
        storage = await caches.open("evodiff-v1");
        hit = await storage.match(url);
        if (hit) return hit.arrayBuffer();
      } else if (
        cache === "opfs" &&
        globalThis.navigator?.storage?.getDirectory
      ) {
        storage = await (
          await navigator.storage.getDirectory()
        ).getDirectoryHandle("evodiff-v1", { create: true });
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
export async function loadWeights(baseUrl, options = {}) {
  if (typeof baseUrl !== "string") throw new TypeError("baseUrl is required");
  const read = createReader(options),
    base = baseUrl.replace(/\/$/, "");
  const manifest = JSON.parse(
    new TextDecoder().decode(await read(base + "/manifest.json")),
  );
  validateManifest(manifest);
  if (
    options.expectedPrecision &&
    manifest.precision !== options.expectedPrecision
  )
    throw new Error("Bundle precision mismatch");
  const weights = {};
  let loaded = 0,
    completed = 0;
  for (const shard of manifest.shards) {
    options.signal?.throwIfAborted();
    const data = await read(base + "/" + shard.file);
    if (
      data.byteLength !== shard.byteLength ||
      (await sha256(data)) !== shard.sha256
    )
      throw new Error(`Shard size/checksum mismatch: ${shard.file}`);
    const parsed = parseSafetensors(data);
    for (const [name, r] of Object.entries(manifest.tensors))
      if (r.shard === shard.file) {
        const t = parsed[name];
        if (
          !t ||
          t.dtype !== r.dtype ||
          t.shape.toString() !== r.shape.toString()
        )
          throw new Error(`Tensor mismatch: ${name}`);
        if (t.dtype === "I8") {
          const scale = parsed[r.quant.scale_tensor];
          if (
            !scale ||
            scale.dtype !== "F32" ||
            scale.shape.toString() !== String(t.shape[0])
          )
            throw new Error("Missing colocated scale");
          t.scales = decodeTensor(scale);
          if (t.scales.some((x) => !Number.isFinite(x) || x <= 0))
            throw new Error("Invalid scales");
        }
        if (options.onTensor) await options.onTensor(name, t);
        else weights[name] = t;
        completed++;
      }
    loaded += data.byteLength;
    options.onProgress?.({
      phase: "weights",
      loaded,
      total: manifest.total_bytes,
    });
  }
  options.signal?.throwIfAborted();
  if (completed !== 199) throw new Error("Incomplete weights");
  return { manifest, weights };
}
