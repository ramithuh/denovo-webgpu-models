import { loadBundle, toFloat, abort } from './loader.js';
import { LaProteinaNetwork } from './network.js';
import { sample, randomNormal, validateLength, validateArray } from './sampling.js';
import { structure } from './pdb.js';
import { extractStructureForAE, extractMotif, computeMotifRMSD } from './motif.js';

const VARIANT_ALIASES = {
  ld1: 'ld1',
  unconditional: 'ld1',
  ld4: 'ld4',
  motif_idx_aa: 'ld4',
  ld5: 'ld5',
  motif_idx_tip: 'ld5',
  ld6: 'ld6',
  motif_uidx_aa: 'ld6',
  ld7: 'ld7',
  motif_uidx_tip: 'ld7',
  ae3: 'ae3',
};

const VARIANT_BUNDLES = {
  ld1: ['LD1', 'AE1'],
  ld4: ['LD4', 'AE3'],
  ld5: ['LD5', 'AE3'],
  ld6: ['LD6', 'AE3'],
  ld7: ['LD7', 'AE3'],
  ae3: ['AE3'],
};

function normalizeVariant(v) {
  const key = VARIANT_ALIASES[String(v ?? 'ld1').toLowerCase()];
  if (!key) throw Error(`Unsupported La-Proteina variant: ${v}`);
  return key;
}

export async function makeEngine(ops, options = {}) {
  const { baseUrl, precision = 'fp32', cache = 'none', read, signal, onProgress } = options;
  if (!baseUrl) throw Error('baseUrl is required');
  if (!['fp32', 'fp16', 'int8'].includes(precision)) throw Error('Unsupported precision');

  const requestedVariants = options.variants
    ? (options.variants === 'all_motif' ? ['ld4', 'ld5', 'ld6', 'ld7'] : options.variants.map(normalizeVariant))
    : [normalizeVariant(options.variant ?? 'ld1')];
  if (!requestedVariants.length) throw Error('At least one variant is required');
  const defaultVariant = requestedVariants[0];

  const bundlesNeeded = [];
  for (const v of requestedVariants) {
    for (const b of VARIANT_BUNDLES[v]) {
      if (!bundlesNeeded.includes(b)) bundlesNeeded.push(b);
    }
  }

  let destroyed = false, busy = false;
  const models = {};
  const manifests = {};
  try {
    const cleanBase = String(baseUrl).replace(/\/$/, '');
    for (const name of bundlesNeeded) {
      const bundle = await loadBundle({
        baseUrl: `${cleanBase}/${name.toLowerCase()}/${precision}`,
        cache,
        read,
        signal,
        onProgress,
      });
      if (bundle.manifest.model !== name || bundle.manifest.precision !== precision) {
        throw Error('Manifest model or precision mismatch');
      }
      const weights = {};
      for (const [k, v] of Object.entries(bundle.weights)) {
        abort(signal);
        weights[k] = ops.uploadWeight ? ops.uploadWeight(v) : ops.upload(v.shape, toFloat(v), true);
        delete bundle.weights[k];
      }
      manifests[name] = bundle.manifest;
      if (name === 'AE3') {
        const encWeights = {};
        const decWeights = {};
        for (const [k, v] of Object.entries(weights)) {
          if (k.startsWith('encoder.')) encWeights[k.slice(8)] = v;
          else if (k.startsWith('decoder.')) decWeights[k.slice(8)] = v;
        }
        models.AE3_encoder = new LaProteinaNetwork(ops, encWeights, bundle.manifest.config, 'AE3_encoder');
        models.AE3_decoder = new LaProteinaNetwork(ops, decWeights, bundle.manifest.config, 'AE3_decoder');
      } else {
        models[name] = new LaProteinaNetwork(ops, weights, bundle.manifest.config, name);
      }
    }
  } catch (e) {
    ops.destroy();
    throw e;
  }

  function resolvePair(variantOpt) {
    const v = variantOpt ? normalizeVariant(variantOpt) : defaultVariant;
    if (!requestedVariants.includes(v)) throw Error(`Variant ${v} is not loaded on this engine (loaded: ${requestedVariants.join(', ')})`);
    if (v === 'ld1') return { variant: v, ld: models.LD1, dec: models.AE1, enc: null };
    if (v === 'ae3') return { variant: v, ld: null, dec: models.AE3_decoder, enc: models.AE3_encoder };
    const ldName = v.toUpperCase();
    return { variant: v, ld: models[ldName], dec: models.AE3_decoder, enc: models.AE3_encoder };
  }

  async function run(fn, opts = {}) {
    if (destroyed) throw Error('Engine destroyed');
    if (busy) throw Error('Concurrent inference on one engine is unsupported');
    busy = true;
    const start = performance.now();
    try {
      abort(opts.signal);
      const result = await fn();
      abort(opts.signal);
      return { ...result, timingMs: performance.now() - start };
    } finally {
      ops.clear();
      busy = false;
    }
  }

  return {
    precision,
    variant: defaultVariant,
    variants: requestedVariants,
    manifests,
    sample(opts = {}) {
      return run(async () => {
        const { variant: v, ld, dec } = resolvePair(opts.variant);
        if (!ld) throw Error(`Variant ${v} has no latent diffusion network for sample(); use encode/decode/encodeDecode`);
        if (v === 'ld1') {
          return structure(await sample(ops, ld, dec, opts));
        }
        if (!opts.motif) throw Error(`Motif variant ${v} requires opts.motif`);
        const motifCond = opts.motif.xMotif && opts.motif.motifMask && opts.motif.seqMotif && opts.motif.activeResidueIndices
          ? opts.motif
          : extractMotif({
              atomSelectionMode: ld.c.motif_atom_mode ?? 'all_atom',
              seed: opts.seed ?? 5,
              ...opts.motif,
            });
        const motifState = ld.prepareMotif(motifCond);
        const raw = await sample(ops, ld, dec, {
          ...opts,
          length: motifCond.length,
          motifState,
          centerEveryStep: ld.c.center_every_step ?? false,
        });
        const out = structure(raw);
        out.variant = v;
        out.resolvedContig = motifCond.resolvedContig;
        out.motifIndices = motifCond.motifIndices;
        out.motifMetrics = computeMotifRMSD(out, motifCond, {
          unindexed: ld.c.motif_indexing === 'unindexed',
        });
        return out;
      }, opts);
    },
    denoiseStep(input, opts = {}) {
      return run(async () => {
        const { variant: v, ld } = resolvePair(opts.variant ?? input.variant);
        if (!ld) throw Error(`Variant ${v} has no latent diffusion network for denoiseStep()`);
        const L = input.length;
        validateLength(L);
        const x = {
          ca: ops.tensor([L, 3], validateArray(input.ca, L * 3, 'ca')),
          z: ops.tensor([L, 8], validateArray(input.z, L * 8, 'z')),
          tCa: input.tCa,
          tZ: input.tZ,
        };
        for (const t of [x.tCa, x.tZ]) if (!Number.isFinite(t) || t < 0 || t >= 1) throw Error('Time must be in [0, 1)');
        if (input.scCa) x.scCa = ops.tensor([L, 3], validateArray(input.scCa, L * 3, 'scCa'));
        if (input.scZ) x.scZ = ops.tensor([L, 8], validateArray(input.scZ, L * 8, 'scZ'));
        if (v !== 'ld1') {
          if (!input.motif) throw Error(`Motif variant ${v} requires input.motif`);
          const motifCond = extractMotif({
            atomSelectionMode: ld.c.motif_atom_mode ?? 'all_atom',
            length: L,
            ...input.motif,
          });
          x.motifState = ld.prepareMotif(motifCond);
        }
        const trace = opts.returnIntermediates ? {} : undefined;
        const out = await ld.forward(x, { ...opts, trace });
        if (ops.readMany) {
          const [rca, rz] = await ops.readMany([out.ca, out.z]);
          return { ca: rca, z: rz, trace };
        }
        return { ca: await ops.read(out.ca), z: await ops.read(out.z), trace };
      }, opts);
    },
    decode(input, opts = {}) {
      return run(async () => {
        const { dec } = resolvePair(opts.variant ?? input.variant);
        const L = input.length;
        validateLength(L);
        const ca = ops.tensor([L, 3], validateArray(input.ca, L * 3, 'ca'));
        const z = ops.tensor([L, 8], validateArray(input.z, L * 8, 'z'));
        const trace = opts.returnIntermediates ? {} : undefined;
        const out = await dec.forward({ ca, z }, { ...opts, trace });
        if (ops.readMany) {
          const [rcoords, rlogits, rca, rz] = await ops.readMany([out.coords, out.logits, ca, z]);
          return structure({ coords: rcoords, logits: rlogits, ca: rca, z: rz, trace });
        }
        return structure({
          coords: await ops.read(out.coords),
          logits: await ops.read(out.logits),
          ca: await ops.read(ca),
          z: await ops.read(z),
          trace,
        });
      }, opts);
    },
    encode(input, opts = {}) {
      return run(async () => {
        const enc = models.AE3_encoder;
        if (!enc) throw Error('AE3 encoder is not loaded on this engine (pass variant: "ae3" or "ld4".."ld7")');
        const st = extractStructureForAE(input, { centerCa: input?.centerCa ?? true });
        const L = st.length;
        const deterministic = Boolean(input?.deterministic ?? opts.deterministic);
        const epsArr = deterministic
          ? null
          : input?.eps
            ? validateArray(input.eps, L * 8, 'eps')
            : randomNormal(input?.seed ?? opts.seed ?? 5)(L * 8);
        const x = {
          coordsNm: ops.tensor([L, 37, 3], st.coordsNm),
          coordMask: ops.tensor([L, 37], st.coordMask),
          residueTypes: ops.tensor([L], st.residueTypes),
          chainBreaks: ops.tensor([L], st.chainBreaks),
          eps: epsArr ? ops.tensor([L, 8], epsArr) : null,
        };
        const trace = opts.returnIntermediates ? {} : undefined;
        const out = await enc.forward(x, { ...opts, trace });
        const [rmean, rlogScale, rz] = ops.readMany
          ? await ops.readMany([out.mean, out.logScale, out.z])
          : [await ops.read(out.mean), await ops.read(out.logScale), await ops.read(out.z)];
        return { length: L, ca: st.ca, mean: rmean, logScale: rlogScale, z: rz, trace };
      }, opts);
    },
    encodeDecode(input, opts = {}) {
      return run(async () => {
        const enc = models.AE3_encoder;
        const dec = models.AE3_decoder;
        if (!enc || !dec) throw Error('AE3 encoder+decoder are not loaded on this engine');
        const st = extractStructureForAE(input, { centerCa: input?.centerCa ?? true });
        const L = st.length;
        const deterministic = Boolean(input?.deterministic ?? opts.deterministic);
        const epsArr = deterministic
          ? null
          : input?.eps
            ? validateArray(input.eps, L * 8, 'eps')
            : randomNormal(input?.seed ?? opts.seed ?? 5)(L * 8);
        const caArr = input?.ca ? Float32Array.from(validateArray(input.ca, L * 3, 'ca')) : st.ca;
        const ca = ops.tensor([L, 3], caArr);
        const x = {
          coordsNm: ops.tensor([L, 37, 3], st.coordsNm),
          coordMask: ops.tensor([L, 37], st.coordMask),
          residueTypes: ops.tensor([L], st.residueTypes),
          chainBreaks: ops.tensor([L], st.chainBreaks),
          eps: epsArr ? ops.tensor([L, 8], epsArr) : null,
          keep: [ca],
        };
        const traceEnc = opts.returnIntermediates ? {} : undefined;
        const traceDec = opts.returnIntermediates ? {} : undefined;
        const encOut = await enc.forward(x, { ...opts, trace: traceEnc });
        const decOut = await dec.forward({ ca, z: encOut.z, keep: [encOut.mean, encOut.logScale] }, { ...opts, trace: traceDec });
        const [rcoords, rlogits, rca, rz, rmean, rlogScale] = ops.readMany
          ? await ops.readMany([decOut.coords, decOut.logits, ca, encOut.z, encOut.mean, encOut.logScale])
          : [
              await ops.read(decOut.coords),
              await ops.read(decOut.logits),
              await ops.read(ca),
              await ops.read(encOut.z),
              await ops.read(encOut.mean),
              await ops.read(encOut.logScale),
            ];
        const trace = opts.returnIntermediates ? { encoder: traceEnc, decoder: traceDec } : undefined;
        return structure({ coords: rcoords, logits: rlogits, ca: rca, z: rz, mean: rmean, logScale: rlogScale, trace });
      }, opts);
    },
    memoryUsage: () => ops.memoryUsage(),
    destroy() {
      if (destroyed) return;
      destroyed = true;
      ops.destroy();
    },
  };
}
