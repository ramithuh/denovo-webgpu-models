/** Safetensors decoding and browser fetch/cache. No Node dependencies. */
export function halfToFloat(h) {
  const s = h & 32768 ? -1 : 1,
    e = (h >> 10) & 31,
    m = h & 1023;
  return e === 0
    ? s * m * 2 ** -24
    : e === 31
      ? m
        ? NaN
        : s * Infinity
      : s * (1 + m / 1024) * 2 ** (e - 15);
}
export function decodeSafetensors(buffer) {
  const view = new DataView(buffer);
  const n = Number(view.getBigUint64(0, true));
  if (!Number.isSafeInteger(n) || n + 8 > buffer.byteLength)
    throw Error("Invalid safetensors header");
  const meta = JSON.parse(
      new TextDecoder().decode(new Uint8Array(buffer, 8, n)),
    ),
    out = {};
  for (const [name, r] of Object.entries(meta)) {
    if (name === "__metadata__") continue;
    const len = r.shape.reduce((a, b) => a * b, 1),
      data = new Float32Array(len),
      start = 8 + n + r.data_offsets[0],
      bytes = { F32: 4, F16: 2, I8: 1 }[r.dtype];
    if (
      !bytes ||
      r.data_offsets[1] - r.data_offsets[0] !== len * bytes ||
      start + len * bytes > buffer.byteLength
    )
      throw Error("Invalid tensor " + name);
    for (let i = 0; i < len; i++)
      data[i] =
        r.dtype === "F32"
          ? view.getFloat32(start + 4 * i, true)
          : r.dtype === "F16"
            ? halfToFloat(view.getUint16(start + 2 * i, true))
            : view.getInt8(start + i);
    out[name] = { data, shape: r.shape };
  }
  return out;
}
export function assembleWeights(manifest, shards) {
  const weights = {};
  for (const [k, r] of Object.entries(manifest.tensors)) {
    const t = shards[r.shard]?.[k];
    if (!t || JSON.stringify(t.shape) !== JSON.stringify(r.shape))
      throw Error("Missing/mismatched weight " + k);
    let data = t.data;
    if (r.quant) {
      if (r.quant.scheme !== "symmetric_per_row" || r.quant.axis !== 0)
        throw Error("Unsupported quantization");
      const scale = shards[r.shard][r.quant.scale_tensor];
      data = Float32Array.from(
        data,
        (v, i) => v * scale.data[Math.floor(i / r.shape[1])],
      );
    }
    weights[k] = { data, shape: t.shape };
  }
  return { manifest, weights };
}
export async function fetchWeights(
  baseUrl,
  { signal, onProgress, cache = true } = {},
) {
  const base = baseUrl.replace(/\/$/, "");
  const response = await fetch(base + "/manifest.json", { signal });
  if (!response.ok) throw Error("Weight manifest HTTP " + response.status);
  const manifest = await response.json();
  if (manifest.format !== "ligandmpnn-safetensors-v1")
    throw Error("Unsupported weight format");
  let store;
  try {
    if (cache && globalThis.caches) store = await caches.open("ligandmpnn-v1");
  } catch {}
  const shards = {};
  let loaded = 0;
  for (const s of manifest.shards) {
    signal?.throwIfAborted();
    const url = base + "/" + s.file;
    const key =
      new URL(url, globalThis.location?.href).href + "?sha256=" + s.sha256;
    let r = await store?.match(key);
    if (!r) {
      r = await fetch(url, { signal });
      if (!r.ok) throw Error("Weight shard HTTP " + r.status);
    }
    const raw = await r.arrayBuffer();
    if (raw.byteLength !== s.bytes) throw Error("Weight size mismatch");
    if (globalThis.crypto?.subtle) {
      const digest = await crypto.subtle.digest("SHA-256", raw);
      const hex = [...new Uint8Array(digest)]
        .map((x) => x.toString(16).padStart(2, "0"))
        .join("");
      if (hex !== s.sha256) {
        await store?.delete(key);
        throw Error("Weight checksum mismatch");
      }
    }
    if (store)
      try {
        await store.put(key, new Response(raw));
      } catch {}
    shards[s.file] = decodeSafetensors(raw);
    loaded += s.bytes;
    onProgress?.({ stage: "weights", loaded, total: manifest.total_bytes });
  }
  return assembleWeights(manifest, shards);
}
