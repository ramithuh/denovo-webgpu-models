import { readShard, parseSafetensors } from './shard.js';
export function validateManifest(manifest) {
  if (manifest?.format !== 'rfd3-packed-v1' || !['fp32', 'fp16', 'int8'].includes(manifest.precision) ||
      !Array.isArray(manifest.shards) || !manifest.tensors || typeof manifest.tensors !== 'object')
    throw new Error('Invalid RFD3 weight manifest');
  const shards = new Map();
  for (const shard of manifest.shards) {
    if (!/^[a-f0-9]{64}\.safetensors$/.test(shard.file) || shards.has(shard.file) ||
        shard.file !== shard.sha256 + '.safetensors' || !Number.isSafeInteger(shard.byteLength) ||
        shard.byteLength <= 0 || shard.byteLength >= 25_000_000) throw new Error('Invalid manifest shard');
    shards.set(shard.file, shard);
  }
  for (const [name, t] of Object.entries(manifest.tensors)) {
    if (!shards.has(t.shard) || t.key !== name || !['F32', 'F16', 'I8'].includes(t.dtype) ||
        !Array.isArray(t.shape) || !t.shape.every(n => Number.isSafeInteger(n) && n > 0) ||
        !Number.isSafeInteger(t.shape.reduce((a, b) => a * b, 1))) throw new Error('Invalid tensor manifest: ' + name);
    if (t.dtype !== 'F32' && (t.role !== 'linear' || t.shape.length !== 2)) throw new Error('Only linear weights may be packed');
    if (t.dtype === 'I8' && (t.scale_key !== name + '.__row_scale' || t.scale_axis !== 0))
      throw new Error('Invalid row quantization: ' + name);
  }
  return manifest;
}

export function normalizeBaseURL(baseURL) {
  const href = baseURL instanceof URL ? baseURL.href : String(baseURL);
  return href.endsWith('/') ? href : href + '/';
}

// Streams shards in manifest order. Consumers copy/upload each tensor so only one
// shard needs to be retained on the host. Even a selective read verifies the whole shard.
export async function* iterateWeights(manifest, baseURL, { names, onProgress, ...options } = {}) {
  validateManifest(manifest);
  const base = normalizeBaseURL(baseURL);
  const wanted = names ? new Set(names) : new Set(Object.keys(manifest.tensors));
  for (const name of wanted) if (!Object.hasOwn(manifest.tensors, name)) throw new Error('Unknown weight: ' + name);
  const active = manifest.shards.map(shard => ({
    shard,
    selected: [...wanted].filter(name => manifest.tensors[name].shard === shard.file),
  })).filter(entry => entry.selected.length > 0);
  const total = active.reduce((sum, entry) => sum + entry.shard.byteLength, 0);
  let loaded = 0;
  for (const { shard, selected } of active) {
    onProgress?.({ phase: 'weights', loaded, total, shard: shard.file });
    const bytes = await readShard(new URL(shard.file, base).href, shard, options);
    loaded += shard.byteLength;
    onProgress?.({ phase: 'weights', loaded, total, shard: shard.file });
    const tensors = parseSafetensors(bytes);
    for (const name of selected) {
      const entry = manifest.tensors[name], tensor = tensors.get(entry.key);
      if (!tensor || tensor.dtype !== entry.dtype || JSON.stringify(tensor.shape) !== JSON.stringify(entry.shape))
        throw new Error('Shard tensor does not match manifest: ' + name);
      const scales = entry.scale_key ? tensors.get(entry.scale_key) : undefined;
      if (entry.scale_key && (!scales || scales.dtype !== 'F32' || scales.shape.length !== 1 || scales.shape[0] !== tensor.shape[0]))
        throw new Error('Invalid scale tensor: ' + name);
      yield { name, ...tensor, scales: scales?.bytes };
    }
  }
}

export function decodeF16(bits) {
  const sign = bits & 0x8000 ? -1 : 1, exponent = (bits >>> 10) & 31, fraction = bits & 1023;
  return sign * (exponent === 0 ? fraction * 2 ** -24 : exponent === 31 ? (fraction ? NaN : Infinity) : (1 + fraction / 1024) * 2 ** (exponent - 15));
}
export function copyF32(bytes) {
  // Safetensors entries need not start on a four-byte boundary.
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const data = new Float32Array(bytes.byteLength / 4);
  for (let i = 0; i < data.length; i++) data[i] = view.getFloat32(4 * i, true);
  return data;
}
export function decodeWeight(tensor) {
  if (tensor.dtype === 'F32') return copyF32(tensor.bytes);
  const count = tensor.shape.reduce((a, b) => a * b, 1), out = new Float32Array(count);
  const view = new DataView(tensor.bytes.buffer, tensor.bytes.byteOffset, tensor.bytes.byteLength);
  const scales = tensor.scales ? copyF32(tensor.scales) : undefined;
  for (let i = 0; i < count; i++) out[i] = tensor.dtype === 'F16' ? decodeF16(view.getUint16(i * 2, true)) :
    view.getInt8(i) * scales[Math.floor(i / tensor.shape[1])];
  return out;
}
