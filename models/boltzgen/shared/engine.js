import {
  loadBundle,
  createReader,
  parseSafetensors,
  decodeTensor,
} from "./loader.js";
import { createFeatures, derivedFeatures, validateFeatures } from "./input.js";
import { randomInputs } from "./random.js";
import { BoltzGen } from "./model.js";
export async function loadEngine(
  backend,
  {
    baseUrl,
    precision,
    read,
    cache = "none",
    signal,
    onProgress,
    disposeDevice = () => {},
  } = {},
) {
  const weights = new Map(),
    lifetime = new AbortController();
  let destroyed = false,
    running = false;
  const start = performance.now();
  try {
    const manifest = await loadBundle(baseUrl, {
      read,
      cache,
      signal,
      onProgress,
      onTensor: async (name, t, scale) =>
        weights.set(
          name,
          backend.uploadPacked
            ? await backend.uploadPacked(t, scale)
            : backend.upload(
                decodeTensor(t, scale ? decodeTensor(scale) : null),
                t.shape,
              ),
        ),
    });
    if (precision && manifest.precision !== precision)
      throw Error("Precision does not match manifest");
    await backend.flush?.(true);
    const model = new BoltzGen(backend, weights);
    async function execute(features, random, options) {
      if (destroyed) throw Error("Engine destroyed");
      if (running) throw Error("Concurrent runs require separate engines");
      running = true;
      const {
        numSteps = options.steps ?? 3,
        recyclingSteps = 0,
        onProgress,
        onStage,
      } = options;
      try {
        const signal = AbortSignal.any([
          lifetime.signal,
          ...(options.signal ? [options.signal] : []),
        ]);
        signal?.throwIfAborted();
        if (!Number.isSafeInteger(numSteps) || numSteps < 1)
          throw Error("numSteps must be a positive integer");
        if (!Number.isSafeInteger(recyclingSteps) || recyclingSteps < 0)
          throw Error("recyclingSteps must be a nonnegative integer");
        const { A } = validateFeatures(features);
        const f = {};
        for (const [k, t] of features)
          if (k.startsWith("f:"))
            f[k.slice(2)] = backend.upload(t.data, t.shape, false);
        for (const [k, t] of Object.entries(derivedFeatures(features)))
          f[k] = backend.upload(t.data, t.shape, false);
        random ??= randomInputs(
          f.atom_pad_mask.shape[1],
          numSteps,
          options.seed ?? 42,
        );
        const rng = Array.from({ length: 1 + 3 * numSteps }, (_, i) => {
          const t = random.get("rng:" + i);
          if (!t) throw Error(`Missing random draw ${i}`);
          const shape =
            i === 0 || i % 3 === 0
              ? [1, A, 3]
              : i % 3 === 1
                ? [1, 4]
                : [1, 1, 3];
          if (
            String(t.shape) !== String(shape) ||
            t.data.length !== shape.reduce((a, b) => a * b, 1) ||
            !t.data.every(Number.isFinite)
          )
            throw Error(`Invalid random draw ${i}`);
          return backend.upload(t.data, t.shape, false);
        });
        const stage = onStage
          ? async (name, ts) => {
              const list = Array.isArray(ts) ? ts : [ts];
              const values = await Promise.all(
                list.map(async (t) => ({
                  shape: t.shape,
                  data: await backend.read(t),
                })),
              );
              signal.throwIfAborted();
              await onStage(name, Array.isArray(ts) ? values : values[0]);
              signal.throwIfAborted();
            }
          : undefined;
        const result = await model.forward(f, rng, {
          numSteps,
          recyclingSteps,
          signal,
          onProgress,
          stage,
        });
        signal.throwIfAborted();
        const outputs = {},
          structure = {};
        for (const name of ["atom_coords", "trajectory", "denoised"]) {
          const list = Array.isArray(result[name])
            ? result[name]
            : [result[name]];
          const refs = [];
          for (let i = 0; i < list.length; i++) {
            const key = name === "atom_coords" ? name : `${name}.${i}`;
            signal.throwIfAborted();
            outputs[key] = await backend.read(list[i]);
            refs.push({ ref: key });
          }
          structure[name] = name === "atom_coords" ? refs[0] : refs;
        }
        return {
          outputs,
          structure,
          timings: result.timings,
          metadata: features.metadata,
        };
      } finally {
        running = false;
        backend.clearTransient();
      }
    }
    return {
      manifest,
      coldLoadMs: performance.now() - start,
      memoryUsage: () => backend.memoryUsage(),
      destroy() {
        if (!destroyed) {
          destroyed = true;
          lifetime.abort();
          backend.destroy();
          weights.clear();
          disposeDevice();
        }
      },
      async generate(options = {}) {
        if (destroyed) throw Error("Engine destroyed");
        const features = createFeatures(options),
          result = await execute(features, null, options),
          mask = features.get("f:atom_pad_mask").data;
        const trim = (r) => {
          const raw = result.outputs[r.ref],
            out = new Float32Array(features.metadata.nAtoms * 3);
          let j = 0;
          for (let a = 0; a < mask.length; a++)
            if (mask[a]) {
              out.set(raw.subarray(a * 3, a * 3 + 3), j);
              j += 3;
            }
          return out;
        };
        return {
          atom_coords: trim(result.structure.atom_coords),
          trajectory: result.structure.trajectory.map(trim),
          denoised: result.structure.denoised.map(trim),
          ...features.metadata,
          timings: result.timings,
        };
      },
      async run(options = {}) {
        if (destroyed) throw Error("Engine destroyed");
        const source = createReader({ read, cache, signal: options.signal });
        let features = options.features,
          random = options.random;
        if (!features) {
          if (!options.inputsUrl)
            throw Error("run requires features or inputsUrl");
          const ts = parseSafetensors(await source(options.inputsUrl));
          features = new Map();
          random = new Map();
          for (const [k, t] of Object.entries(ts)) {
            const v = { shape: t.shape, data: decodeTensor(t) };
            if (k.startsWith("f:")) features.set(k, v);
            else if (k.startsWith("rng:")) random.set(k, v);
          }
        }
        return execute(features, random, options);
      },
    };
  } catch (error) {
    backend.destroy();
    disposeDevice();
    throw error;
  }
}
