import { loadWeights, decodeTensor } from "./weights.js";
import { buildFeatures, sampleScaffold } from "./features.js";
import {
  diffusionSchedule,
  gaussianNoise,
  seededRandom,
  reverseStep,
} from "./diffusion.js";
import { frenet } from "./geometry.js";
import { denoise } from "./network.js";
export async function initializeEngine(o, options, metadata = {}) {
  const {
    baseUrl,
    precision = "fp32",
    read,
    cache = "none",
    signal,
    onProgress,
  } = options;
  if (!["fp32", "fp16", "int8"].includes(precision))
    throw Error("Unsupported precision");
  let destroyed = false,
    busy = false;
  const weights = {};
  let bundle;
  try {
    bundle = await loadWeights(baseUrl, {
      read,
      cache,
      signal,
      onProgress,
      onTensor: (name, t) => {
        weights[name] = o.uploadWeight
          ? o.uploadWeight(t)
          : o.upload(t.shape, decodeTensor(t), true);
      },
    });
    if (bundle.manifest.precision !== precision)
      throw Error("Precision does not match bundle");
  } catch (e) {
    o.destroy();
    metadata.release?.();
    throw e;
  }
  const c = bundle.config;
  const uploadFeatures = (f) =>
    Object.fromEntries(
      Object.entries(f)
        .filter(([k]) => k !== "length")
        .map(([k, v]) => [
          k,
          o.tensor(
            k === "aatype"
              ? [f.length, 20]
              : k === "atom_positions"
                ? [f.length, 3]
                : k === "fixed_structure_mask"
                  ? [f.length, f.length]
                  : [f.length],
            v,
          ),
        ]),
    );
  const run = async (fn) => {
    if (destroyed) throw Error("Engine destroyed");
    if (busy) throw Error("Engine is already running");
    busy = true;
    try {
      return await fn();
    } finally {
      try {
        o.clear();
      } finally {
        busy = false;
      }
    }
  };
  const engine = {
    precision,
    config: c,
    ...metadata,
    manifest: bundle.manifest,
    async forward({
      coordinates,
      rotations,
      features,
      length = coordinates?.length / 3,
      timestep = 1000,
      steps = 1000,
      signal,
      onStage,
    } = {}) {
      return run(async () => {
        signal?.throwIfAborted();
        diffusionSchedule(steps);
        if (!Number.isInteger(timestep) || timestep < 1 || timestep > steps)
          throw new RangeError("timestep must be in [1, steps]");
        const f = buildFeatures({ length, ...features }),
          L = f.length;
        if (
          coordinates?.length !== L * 3 ||
          Array.from(coordinates).some((x) => !Number.isFinite(x))
        )
          throw Error("Invalid input coordinates");
        const ft = uploadFeatures(f),
          trans = o.tensor([L, 3], coordinates);
        let rots;
        if (rotations) {
          if (
            rotations.length !== L * 9 ||
            Array.from(rotations).some((x) => !Number.isFinite(x))
          )
            throw Error("Invalid rotation shape or values");
          rots = o.tensor([L, 3, 3], rotations);
        } else rots = frenet(o, trans, ft.chain_index, ft.residue_mask);
        const out = await denoise(
          o,
          weights,
          c,
          { rots, trans },
          ft,
          timestep,
          steps,
          {
            signal,
            onStage: onStage
              ? async (n, x) => onStage(n, await o.read(x))
              : undefined,
          },
        );
        return {
          z: await o.read(out.z),
          coordinates: await o.read(out.trans),
          rotations: await o.read(out.rots),
        };
      });
    },
    async sample({
      length = 128,
      features,
      motifs,
      scaffold,
      steps = 1000,
      scale = 0.6,
      seed = 1,
      initialNoise,
      noises,
      signal,
      onStep,
      onProgress,
      onStage,
      readStepCoordinates = true,
      trajectory = false,
    } = {}) {
      return run(async () => {
        signal?.throwIfAborted();
        if (!Number.isFinite(scale) || scale < 0 || scale > 1)
          throw new RangeError("scale must be in [0,1]");
        const schedule = diffusionSchedule(steps),
          rng = seededRandom(seed),
          spec = scaffold ? sampleScaffold(scaffold, rng) : { length, motifs };
        const f = buildFeatures({ ...spec, ...features }),
          L = f.length,
          ft = uploadFeatures(f),
          saved = [];
        const noiseTensor = (data) => {
          if (
            data.length !== L * 3 ||
            Array.from(data).some((x) => !Number.isFinite(x))
          )
            throw Error("Invalid exact noise draw");
          return o.tensor([L, 3], data);
        };
        let trans = o.mul(
          noiseTensor(initialNoise ?? gaussianNoise(L * 3, rng)),
          o.reshape(ft.residue_mask, [L, 1]),
        );
        for (let t = steps; t >= 1; t--) {
          signal?.throwIfAborted();
          const rots = frenet(o, trans, ft.chain_index, ft.residue_mask);
          const out = await denoise(
            o,
            weights,
            c,
            { rots, trans },
            ft,
            t,
            steps,
            {
              signal,
              onStage: onStage
                ? async (n, x) => onStage(t, n, await o.read(x))
                : undefined,
            },
          );
          const noise = noiseTensor(
            noises?.[steps - t] ??
              (t > 1 ? gaussianNoise(L * 3, rng) : new Float32Array(L * 3)),
          );
          trans = reverseStep(
            o,
            trans,
            out.z,
            noise,
            ft.residue_mask,
            schedule,
            t,
            scale,
          );
          if (trajectory || (onStep && readStepCoordinates)) {
            const xyz = await o.read(trans);
            if (trajectory) saved.push(xyz);
            await onStep?.({ step: t, coordinates: xyz });
          } else if (onStep) {
            await onStep({ step: t });
          }
          onProgress?.({ step: steps - t + 1, totalSteps: steps, t });
          await o.checkpoint([...Object.values(ft), trans], {
            sync: t === 1 || t % 5 === 0,
          });
          signal?.throwIfAborted();
        }
        const rotations = frenet(o, trans, ft.chain_index, ft.residue_mask);
        return {
          coordinates: await o.read(trans),
          rotations: await o.read(rotations),
          features: f,
          trajectory: saved,
          steps,
          scale,
          seed,
        };
      });
    },
    memoryUsage() {
      return o.memoryUsage();
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      o.destroy();
      for (const key of Object.keys(weights)) delete weights[key];
      metadata.release?.();
    },
  };
  delete engine.release;
  return engine;
}
