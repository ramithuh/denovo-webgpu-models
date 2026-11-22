import { LaserModel } from "./model.js";
import { parseStructure, featurize, writePdb } from "./structure.js";
import { makeGeometry, extend, radians } from "./geometry.js";
function logProb(logits, index) {
  const m = Math.max(...logits);
  return (
    logits[index] -
    m -
    Math.log(
      Array.from(logits, (x) => Math.exp(x - m)).reduce((a, b) => a + b, 0),
    )
  );
}
export function createDesignEngine(
  backend,
  weights,
  { onProgress, signal, onDestroy } = {},
) {
  let destroyed = false,
    busy = false;
  const model = new LaserModel(backend, weights.manifest.config),
    geometry = makeGeometry(weights.tensors);
  return {
    manifest: weights.manifest,
    async design(structureText, opts = {}) {
      if (destroyed) throw Error("Engine destroyed");
      if (busy) throw Error("Only one design call may run at a time");
      const options = {
        ...opts,
        signal: opts.signal ?? signal,
        onProgress: opts.onProgress ?? onProgress,
      };
      for (const key of ["temperature", "chiTemperature"])
        if (
          options[key] != null &&
          (!Number.isFinite(options[key]) || options[key] < 0)
        )
          throw Error(`${key} must be nonnegative`);
      if (
        options.seed != null &&
        (!Number.isInteger(options.seed) ||
          options.seed < 0 ||
          options.seed > 0xffffffff)
      )
        throw Error("seed must be a uint32");
      const num = options.numSequences ?? 1;
      if (!Number.isInteger(num) || num < 1)
        throw Error("numSequences must be a positive integer");
      if (
        options.omitAAs != null &&
        (typeof options.omitAAs !== "string" ||
          /[^ARNDCEQGHILKMFPSTWYVX]/.test(options.omitAAs))
      )
        throw Error("omitAAs must contain uppercase amino-acid codes");
      if (new Set((options.omitAAs ?? "") + "X").size === 21)
        throw Error("omitAAs excludes every amino acid");
      options.signal?.throwIfAborted();
      busy = true;
      const mark = backend.mark(),
        start = performance.now();
      try {
        const structure = parseStructure(structureText, options),
          sequences = [],
          scores = [],
          chiAngles = [],
          pdbTexts = [];
        for (const id of options.fixedPositions ?? [])
          if (!structure.residues.some((r) => r.id === id))
            throw Error("Unknown fixed position " + id);
        for (let s = 0; s < num; s++) {
          if (destroyed) throw Error("Engine destroyed");
          const runOptions = { ...options, seed: (options.seed ?? 42) + s },
            batch = featurize(
              structure,
              weights.tensors,
              weights.manifest.config,
              runOptions,
            );
          const result = await model.run(batch, runOptions),
            built = geometry.build(
              batch.backbone,
              result.sequence,
              result.chiAngles,
            );
          // NH is separate from upstream's 24-atom padded sidechain tensor.
          const nh = new Float32Array(batch.length * 3).fill(NaN);
          for (let i = 0; i < batch.length; i++)
            if (result.sequence[i] !== 14) {
              const points = [2, 1, 0].map((a) =>
                Array.from(
                  built.coords.slice(
                    (i * built.atomsPerResidue + a) * 3,
                    (i * built.atomsPerResidue + a + 1) * 3,
                  ),
                ),
              );
              nh.set(
                extend(
                  points,
                  1,
                  radians(120),
                  radians(batch.phiPsi[2 * i] + 180),
                ),
                i * 3,
              );
            }
          built.nhCoords = nh;
          geometry.cleanup(built, result.sequence);
          sequences.push(
            Array.from(
              result.sequence,
              (i) => weights.manifest.config.alphabet[i],
            ).join(""),
          );
          chiAngles.push(result.chiAngles);
          pdbTexts.push(writePdb(batch, result, built));
          let sequenceLogLikelihood = 0,
            chiLogLikelihood = 0;
          const perResidue = [];
          for (let i = 0; i < batch.length; i++) {
            const aa = logProb(
              result.sequenceLogits.subarray(i * 21, i * 21 + 21),
              result.sequence[i],
            );
            let chi = 0;
            for (let c = 0; c < 4; c++)
              if (Number.isFinite(result.chiAngles[i * 4 + c])) {
                const bin = batch.fixed[i]
                  ? ((Math.round((result.chiAngles[i * 4 + c] + 180) / 5) %
                      72) +
                      72) %
                    72
                  : result.chiBins[i * 4 + c];
                chi += logProb(
                  result.chiLogits.subarray(
                    i * 288 + c * 72,
                    i * 288 + (c + 1) * 72,
                  ),
                  bin,
                );
              }
            sequenceLogLikelihood += aa;
            chiLogLikelihood += chi;
            perResidue.push({ sequence: aa, chi });
          }
          scores.push({
            sequenceLogLikelihood,
            chiLogLikelihood,
            totalLogLikelihood: sequenceLogLikelihood + chiLogLikelihood,
            perResidue,
          });
        }
        return {
          sequences,
          scores,
          chiAngles,
          pdbTexts,
          timingMs: performance.now() - start,
        };
      } finally {
        backend.releaseSince(mark);
        busy = false;
      }
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      backend.destroy();
      onDestroy?.();
    },
  };
}
