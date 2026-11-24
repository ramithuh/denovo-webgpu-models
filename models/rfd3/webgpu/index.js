import { validateFeatures } from '../shared/features.js';
import { generate } from '../shared/input.js';
import { sampleGPU } from '../shared/sampler.js';
import { normalizeBaseURL } from '../shared/weights.js';
import { createGpuResources } from './resources.js';
import { loadGPUWeights } from './weights.js';
import { GPUOps } from './ops.js';
import { RFD3GPUModel } from './model.js';

async function readJSON(url, { read, fetcher = fetch, signal } = {}) {
  if (read) {
    const raw = await read(url, { signal });
    if (raw instanceof ArrayBuffer) return JSON.parse(new TextDecoder().decode(raw));
    if (ArrayBuffer.isView(raw)) return JSON.parse(new TextDecoder().decode(raw));
    if (raw && typeof raw === 'object' && 'ok' in raw) {
      if (!raw.ok) throw new Error('JSON fetch failed (HTTP ' + raw.status + '): ' + url);
      return raw.json();
    }
    throw new TypeError('Custom read hook must return ArrayBuffer, TypedArray, or Response');
  }
  const response = await fetcher(url, { signal });
  if (!response.ok) throw new Error('JSON fetch failed (HTTP ' + response.status + '): ' + url);
  return response.json();
}

export async function createRFD3Engine({ manifest, architecture, baseURL, baseUrl, precision, ...options } = {}) {
  const rawBase = baseURL ?? baseUrl;
  let resolvedBase = rawBase != null ? normalizeBaseURL(rawBase) : undefined;
  if (!manifest) {
    if (!resolvedBase) throw new Error('Either manifest or baseURL is required');
    if (precision && !resolvedBase.endsWith('/' + precision + '/'))
      resolvedBase = normalizeBaseURL(new URL(precision + '/', resolvedBase));
    manifest = await readJSON(new URL('manifest.json', resolvedBase).href, options);
  }
  if (!architecture?.config || manifest?.checkpoint_sha256 !== architecture.checkpoint_sha256)
    throw new Error('Architecture/checkpoint mismatch');
  const resources = await createGpuResources(options);
  let store, ops, closed = false, busy = false;
  const handles = new Set();
  try {
    store = await loadGPUWeights({ manifest, baseURL: resolvedBase, resources, ...options });
    ops = new GPUOps(resources);
    const config = { ...architecture.config, sequence_valid_out_mask: manifest.constants['diffusion_module.sequence_head.valid_out_mask'].value };
    const model = new RFD3GPUModel(ops, store.tensors, config);
    const assertLive = () => { if (closed || resources.closed) throw new Error('RFD3 GPU engine is closed'); options.signal?.throwIfAborted(); };
    async function transaction(fn) {
      assertLive();
      if (busy) throw new Error('Concurrent calls on one GPU engine are unsupported');
      busy = true;
      const device = resources.device, before = new Set(ops.tensors);
      device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation');
      let result, failure;
      try { ops.begin(); result = fn(); await ops.submit(); }
      catch (error) { failure = error; ops.encoder = null; }
      try {
        const validation = await device.popErrorScope(), allocation = await device.popErrorScope();
        if (validation || allocation) failure ??= new Error((validation ?? allocation).message);
      } finally { busy = false; }
      if (failure) {
        for (const t of [...ops.tensors]) if (!before.has(t)) ops.free(t);
        if (!resources.closed) ops.trim();
        throw failure;
      }
      return result;
    }
    const engine = {
      precision: manifest.precision, resources,
      async initialize(features) {
        assertLive();
        const { tokens: n, atoms: l } = validateFeatures(features, config);
        const maxBytes = resources.device.limits.maxStorageBufferBindingSize;
        const maxPairBytes = Math.max(l * l * config.c_atompair * 4, n * n * config.c_z * 4);
        if (maxPairBytes > maxBytes)
          throw new RangeError(`Input (${n} tokens, ${l} atoms) requires ${maxPairBytes} bytes, exceeding maxStorageBufferBindingSize (${maxBytes})`);
        let ctx, init;
        await transaction(() => { ctx = model.prepare(features); init = model.initialize(ctx); });
        ops.trim();
        const handle = { ctx, init, engine, released: false }; handles.add(handle); return handle;
      },
      releaseInitializer(handle) {
        if (!handles.delete(handle)) return; handle.released = true;
        for (const t of Object.values(handle.init)) ops.free(t);
        for (const t of Object.values(handle.ctx.f)) ops.free(t);
        for (const key of ['rep', 'downMask', 'upMask', 'atomForced', 'tokenForced']) { for (const t of Object.values(handle.ctx[key]?.multi ?? {})) ops.free(t); ops.free(handle.ctx[key]); }
      },
      async readInitializer(handle) {
        assertLive(); if (!handles.has(handle)) throw new Error('Invalid initializer');
        const { n, l } = handle.ctx, { c_s: cs, c_z: cz, c_atom: ca, c_atompair: cp } = config;
        const shapes = { Q_L_init: [l, ca], C_L: [l, ca], P_LL: [l, l, cp], S_I: [n, cs], Z_II: [n, n, cz] }, out = {};
        for (const [key, t] of Object.entries(handle.init)) out[key] = { shape: shapes[key], data: await ops.read(t) }; return out;
      },
      async inspectNeighbours(handle, coordinates, { token = false } = {}) {
        if (!handles.has(handle)) throw new Error('Invalid initializer'); let input, out;
        try { await transaction(() => { input = ops.upload(coordinates); out = model.neighbours(input, handle.ctx[token ? 'tokenForced' : 'atomForced'], token ? config.diffusion_module.diffusion_transformer.n_keys : config.diffusion_module.n_attn_keys); }); return out ? await ops.read(out) : new Float32Array((coordinates.length / 3) ** 2).fill(1); } finally { if (input) ops.free(input); if (out) ops.free(out); }
      },
      async run({ features, X_noisy_L, t, n_recycle = config.diffusion_module.n_recycle, initializer, signal, random, ...sampling }) {
        if (X_noisy_L === undefined && random) return this.sample({ features, n_recycle, random, signal, ...sampling });
        assertLive(); signal?.throwIfAborted();
        if (!(X_noisy_L instanceof Float32Array) || !X_noisy_L.every(Number.isFinite) || !Number.isInteger(n_recycle) || n_recycle < 1) throw new Error('Invalid denoiser arguments');
        const l = features.atom_to_token_map.length, batch = X_noisy_L.length / (l * 3), times = typeof t === 'number' ? Array(batch).fill(t) : t;
        if (!Number.isInteger(batch) || batch < 1 || times?.length !== batch || !Array.from(times).every(x => Number.isFinite(x) && x >= 0)) throw new Error('Invalid coordinates or times');
        const own = !initializer, handle = initializer ?? await this.initialize(features);
        if (!handles.has(handle) || handle.ctx.host !== features) throw new Error('Initializer does not belong to these features');
        const n = handle.ctx.n, result = { X_L: { shape: [batch, l, 3], data: new Float32Array(batch * l * 3) }, sequence_logits_I: { shape: [batch, n, 32], data: new Float32Array(batch * n * 32) }, sequence_indices_I: { shape: [batch, n], data: new Int32Array(batch * n) } };
        try {
          for (let b = 0; b < batch; b++) {
            signal?.throwIfAborted(); let input, out;
            try {
              out = await transaction(() => { input = ops.upload(X_noisy_L.subarray(b * l * 3, (b + 1) * l * 3)); return model.denoise(handle.ctx, input, times[b], n_recycle, handle.init); });
              for (const [key, value] of Object.entries(out)) { const data = await ops.read(value); result[key].data.set(data, b * data.length); }
            } finally { if (input) ops.free(input); if (out) for (const tensor of Object.values(out)) ops.free(tensor); }
          }
          return result;
        } finally { if (own) this.releaseInitializer(handle); }
      },
      generate(options) {
        return generate(this, { ...options, maxStorageBufferBindingSize: options?.maxStorageBufferBindingSize ?? resources.device.limits.maxStorageBufferBindingSize });
      },
      sample(options) { return sampleGPU(this, options); },
      trim() { assertLive(); if (busy) throw new Error('GPU operation in progress'); ops.trim(); resources.resetPeakMemoryUsage(); },
      memoryUsage: () => resources.memoryUsage(), peakMemoryUsage: () => resources.peakMemoryUsage(),
      destroy() { if (closed) return; closed = true; handles.clear(); ops.destroy(); store.destroy(); resources.destroy(); },
    };
    return engine;
  } catch (error) { ops?.destroy(); store?.destroy(); resources.destroy(); throw error; }
}
export { createRFD3Engine as createRfd3Engine };
