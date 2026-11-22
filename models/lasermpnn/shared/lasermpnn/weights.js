/** Browser/Worker-safe Safetensors reader. All IO is injected or uses fetch. */
export function halfToFloat(h) {
  const s = h & 32768 ? -1 : 1,
    e = (h >>> 10) & 31,
    f = h & 1023;
  return e === 31
    ? f
      ? NaN
      : s * Infinity
    : e === 0
      ? s * 2 ** -24 * f
      : s * 2 ** (e - 15) * (1 + f / 1024);
}
export function parseSafetensors(input) {
  const bytes =
    input instanceof ArrayBuffer
      ? new Uint8Array(input)
      : new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const h = Number(dv.getBigUint64(0, true));
  if (!Number.isSafeInteger(h) || h < 2 || h + 8 > bytes.length)
    throw Error("Invalid Safetensors header");
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + h))),
    result = {};
  for (const [n, r] of Object.entries(header)) {
    if (n === "__metadata__") continue;
    const count = r.shape.reduce((a, b) => a * b, 1),
      size = { F32: 4, I32: 4, F16: 2, I8: 1 }[r.dtype];
    const [start, end] = r.data_offsets;
    if (
      !size ||
      start < 0 ||
      end - start !== count * size ||
      8 + h + end > bytes.length
    )
      throw Error(`Invalid tensor ${n}`);
    const data =
      r.dtype === "I32" ? new Int32Array(count) : new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const at = 8 + h + start + i * size;
      data[i] =
        r.dtype === "F32"
          ? dv.getFloat32(at, true)
          : r.dtype === "I32"
            ? dv.getInt32(at, true)
            : r.dtype === "I8"
              ? dv.getInt8(at)
              : halfToFloat(dv.getUint16(at, true));
    }
    result[n] = { data, shape: r.shape, dtype: r.dtype };
  }
  return result;
}
export async function loadWeights(
  baseUrl,
  { read, signal, onProgress, cache = true } = {},
) {
  const fetchFile =
    read ??
    (async (name) => {
      const url = `${baseUrl.replace(/\/$/, "")}/${name}`;
      let store;
      try {
        if (cache && globalThis.caches)
          store = await caches.open("lasermpnn-v1");
      } catch {}
      let response = await store?.match(url);
      if (!response) {
        response = await fetch(url, { signal });
        if (!response.ok) throw Error(`HTTP ${response.status}: ${url}`);
        if (store) await store.put(url, response.clone());
      }
      return new Uint8Array(await response.arrayBuffer());
    });
  const manifest = JSON.parse(
    new TextDecoder().decode(await fetchFile("manifest.json")),
  );
  if (manifest.format !== "lasermpnn-safetensors-v1")
    throw Error("Unsupported LASErMPNN manifest");
  const tensors = {};
  let done = 0;
  for (const shard of manifest.shards) {
    signal?.throwIfAborted();
    const bytes = await fetchFile(shard.file);
    if (bytes.byteLength !== shard.bytes)
      throw Error(`Truncated shard ${shard.file}`);
    if (globalThis.crypto?.subtle) {
      const digest = await crypto.subtle.digest("SHA-256", bytes);
      const hash = Array.from(new Uint8Array(digest), (x) =>
        x.toString(16).padStart(2, "0"),
      ).join("");
      if (hash !== shard.sha256) throw Error(`Checksum mismatch ${shard.file}`);
    }
    const ts = parseSafetensors(bytes);
    for (const [n, r] of Object.entries(manifest.tensors))
      if (r.shard === shard.file) {
        const t = ts[n];
        if (
          !t ||
          JSON.stringify(t.shape) !== JSON.stringify(r.shape) ||
          t.dtype !== r.dtype
        )
          throw Error(`Manifest mismatch ${n}`);
        if (r.quant) {
          const scale = ts[r.quant.scale_tensor];
          if (
            r.quant.scheme !== "symmetric_per_row" ||
            r.quant.axis !== 0 ||
            !scale ||
            scale.data.length !== t.shape[0]
          )
            throw Error("Invalid row quantization");
          for (let i = 0; i < t.data.length; i++)
            t.data[i] *= scale.data[Math.floor(i / t.shape[1])];
        }
        tensors[n] = t;
      }
    done += shard.bytes;
    onProgress?.({
      phase: "weights",
      loaded: done,
      total: manifest.total_bytes,
    });
  }
  return { manifest, tensors };
}
