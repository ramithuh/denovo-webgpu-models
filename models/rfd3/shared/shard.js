// Cache immutable content by digest. Every read, including a cache hit, is verified.
const maxBytes = 25_000_000;
const namespace = "rfd3-shards-v1";

async function openStore(kind, digest, byteLength, signal) {
  if (kind === "opfs") {
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle(namespace, { create: true });
    return {
      async read() {
        const handle = await dir.getFileHandle(digest);
        const file = await handle.getFile();
        if (file.size !== byteLength) throw new RangeError("Cached shard length mismatch");
        return file.arrayBuffer();
      },
      async write(bytes) {
        const handle = await dir.getFileHandle(digest, { create: true });
        const stream = await handle.createWritable();
        try {
          await stream.write(bytes);
          await stream.close();
        } catch (error) {
          await stream.abort().catch(() => {});
          throw error;
        }
      },
    };
  }
  const cache = await caches.open(namespace);
  const key = new URL("/" + namespace + "/" + digest, location.origin).href;
  return {
    async read() {
      const response = await cache.match(key);
      return response ? readBytes(response, byteLength, signal) : undefined;
    },
    write: bytes => cache.put(key, new Response(bytes)),
  };
}

async function readBytes(response, byteLength, signal) {
  if (!response.body) throw new Error("Missing shard body");
  const reader = response.body.getReader();
  const bytes = new Uint8Array(byteLength);
  let offset = 0;
  const cancel = () => { reader.cancel(signal.reason).catch(() => {}); };
  signal?.addEventListener('abort', cancel, { once: true });
  try {
    signal?.throwIfAborted();
    for (;;) {
      const { done, value } = await abortable(reader.read(), signal);
      if (done) break;
      if (value.length > byteLength - offset) {
        await reader.cancel();
        throw new Error("Shard exceeds declared length");
      }
      bytes.set(value, offset);
      offset += value.length;
    }
  } finally {
    signal?.removeEventListener('abort', cancel);
    reader.releaseLock();
  }
  if (offset !== byteLength) throw new Error("Truncated shard");
  return bytes.buffer;
}

async function valid(bytes, byteLength, sha256) {
  if (bytes?.byteLength !== byteLength) return false;
  const hash = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(hash, value => value.toString(16).padStart(2, "0")).join("") === sha256;
}

export async function readShard(url, { byteLength, sha256 }, {
  cache = "auto", signal, fetcher = fetch, read, openStorage = openStore,
} = {}) {
  if (!Number.isSafeInteger(byteLength) || byteLength <= 0 || byteLength >= maxBytes)
    throw new RangeError("Invalid shard byte length");
  if (typeof sha256 !== "string" || !/^[a-f0-9]{64}$/.test(sha256))
    throw new TypeError("Expected a lowercase SHA-256 digest");
  if (!["auto", "opfs", "cache-storage", "none"].includes(cache))
    throw new TypeError("Unknown cache selection");

  const stores = [];
  const backends = cache === "auto" ? ["opfs", "cache-storage"] : cache === "none" ? [] : [cache];
  for (const backend of backends) {
    signal?.throwIfAborted();
    try {
      const store = await abortable(openStorage(backend, sha256, byteLength, signal), signal);
      stores.push(store);
      const bytes = await abortable(store.read(), signal);
      if (await abortable(valid(bytes, byteLength, sha256), signal)) {
        signal?.throwIfAborted();
        return bytes;
      }
    } catch {
      signal?.throwIfAborted();
      // Private browsing, quota errors, and missing entries permit a fetch.
    }
  }
  signal?.throwIfAborted();
  let bytes;
  if (read) {
    const raw = await abortable(read(url, { signal }), signal);
    if (raw instanceof ArrayBuffer) bytes = raw;
    else if (ArrayBuffer.isView(raw)) bytes = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
    else if (raw && typeof raw === "object" && "ok" in raw) {
      if (!raw.ok) throw new Error("Shard fetch failed (HTTP " + raw.status + ")");
      bytes = await readBytes(raw, byteLength, signal);
    } else throw new TypeError("Custom read hook must return ArrayBuffer, TypedArray, or Response");
    if (bytes.byteLength > byteLength) throw new Error("Shard exceeds declared length");
    if (bytes.byteLength < byteLength) throw new Error("Truncated shard");
  } else {
    const response = await abortable(fetcher(url, { signal, cache: "no-store" }), signal);
    if (!response.ok) throw new Error("Shard fetch failed (HTTP " + response.status + ")");
    bytes = await readBytes(response, byteLength, signal);
  }
  signal?.throwIfAborted();
  if (!await abortable(valid(bytes, byteLength, sha256), signal)) throw new Error("Shard integrity check failed");
  for (const store of stores) {
    signal?.throwIfAborted();
    try {
      await abortable(store.write(bytes), signal);
      break;
    } catch {
      signal?.throwIfAborted();
    }
  }
  signal?.throwIfAborted();
  return bytes;
}

// Return byte views so consumers can keep narrow storage types packed.
export function parseSafetensors(bytes) {
  if (!(bytes instanceof ArrayBuffer) || bytes.byteLength < 10 || bytes.byteLength >= maxBytes)
    throw new RangeError("Invalid safetensors shard size");
  const headerLength = new DataView(bytes).getBigUint64(0, true);
  if (headerLength > BigInt(bytes.byteLength - 8)) throw new RangeError("Truncated safetensors header");
  const start = 8 + Number(headerLength);
  if (new Uint8Array(bytes)[8] !== 123) throw new Error('Safetensors header must start with {');
  const header = parseUniqueJSON(new TextDecoder('utf-8', { fatal: true }).decode(
    new Uint8Array(bytes, 8, Number(headerLength))));
  if (!header || Array.isArray(header) || typeof header !== "object") throw new Error("Invalid tensor header");
  const sizes = new Map([["F32", 4], ["F16", 2], ["I8", 1]]);
  const tensors = new Map();
  const ranges = [];
  for (const [name, entry] of Object.entries(header)) {
    if (name === '__metadata__') {
      if (!entry || typeof entry !== 'object' || Array.isArray(entry) ||
          !Object.values(entry).every(v => typeof v === 'string')) throw new Error('Invalid __metadata__');
      continue;
    }
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) throw new Error('Invalid tensor entry');
    const { dtype, shape, data_offsets: offsets } = entry;
    if (!sizes.has(dtype) || !Array.isArray(shape) || !shape.every(n => Number.isSafeInteger(n) && n >= 0))
      throw new Error("Invalid dtype or shape for " + name);
    if (!Array.isArray(offsets) || offsets.length !== 2 || !offsets.every(Number.isSafeInteger))
      throw new Error("Invalid offsets for " + name);
    const [lo, hi] = offsets;
    const length = shape.reduce((count, dimension) => count * dimension, sizes.get(dtype));
    if (!Number.isSafeInteger(length) || lo < 0 || hi < lo || hi > bytes.byteLength - start || hi - lo !== length)
      throw new Error("Tensor span mismatch for " + name);
    ranges.push([lo, hi]);
    tensors.set(name, { dtype, shape, bytes: new Uint8Array(bytes, start + lo, length) });
  }
  ranges.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let end = 0;
  for (const [lo, hi] of ranges) {
    if (lo !== end) throw new Error("Tensor spans overlap or leave a gap");
    end = hi;
  }
  if (start + end !== bytes.byteLength) throw new Error("Unclaimed tensor data");
  return tensors;
}

// Race non-cancellable browser storage operations without leaking listeners. Underlying
// writes may finish after abort, but only already-verified content is ever persisted.
function abortable(promise, signal) {
  if (!signal) return Promise.resolve(promise);
  return new Promise((resolve, reject) => {
    const aborted = () => reject(signal.reason ?? new DOMException('Aborted', 'AbortError'));
    if (signal.aborted) aborted();
    else signal.addEventListener('abort', aborted, { once: true });
    Promise.resolve(promise).then(resolve, reject).finally(() => signal.removeEventListener('abort', aborted));
  });
}

// JSON.parse alone silently accepts duplicate keys. Walk the validated JSON token
// stream, comparing decoded keys (including Unicode escapes) at every object level.
function parseUniqueJSON(text) {
  const result = JSON.parse(text);
  const tokens = text.match(/"(?:[^"\\]|\\.)*"|[{}\[\]:,]|[^\s{}\[\]:,]+/g) ?? [];
  let i = 0;
  function value() {
    const t = tokens[i++];
    if (t === '{') {
      const keys = new Set();
      if (tokens[i] !== '}') for (;;) {
        const key = JSON.parse(tokens[i++]);
        if (keys.has(key)) throw new Error('Duplicate safetensors key: ' + key);
        keys.add(key); i++; value();
        if (tokens[i] !== ',') break;
        i++;
      }
      i++;
    } else if (t === '[') {
      if (tokens[i] !== ']') for (;;) { value(); if (tokens[i] !== ',') break; i++; }
      i++;
    }
  }
  value();
  return result;
}
