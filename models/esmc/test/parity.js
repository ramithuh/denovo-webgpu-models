import { diffStats } from "../shared/weights.js";
import { AA_IDS } from "../shared/input.js";
export function check(ok, message) {
  if (!ok) throw new Error(message);
}
export function oracle(meta, buffer) {
  check(buffer.byteLength === meta.byteLength, "Oracle length mismatch");
  return (name, index) => {
    const r = meta.tensors[name];
    check(!!r, `Missing oracle tensor ${name}`);
    let n = r.length,
      offset = r.offset;
    if (index !== undefined) {
      n /= r.shape[0];
      offset += index * n * 4;
    }
    const b = buffer.slice(offset, offset + n * 4);
    return r.dtype === "I32" ? new Int32Array(b) : new Float32Array(b);
  };
}
export async function verifyProtein(
  engine,
  meta,
  ref,
  { log = () => {}, warm = false } = {},
) {
  const tensors = {},
    start = performance.now();
  let maxRel = 0;
  const compare = (name, a, b) => {
    const stats = diffStats(a, b);
    check(stats.relRms < 1e-4, `${name}: ${JSON.stringify(stats)}`);
    maxRel = Math.max(maxRel, stats.relRms);
    tensors[name] = stats;
  };
  const traces = (prefix, got, index) => {
    // Iterate expected keys so missing traces cannot silently pass.
    for (const key of Object.keys(meta.tensors).filter((k) =>
      k.startsWith(prefix + "/"),
    )) {
      const k = key.slice(prefix.length + 1);
      if (
        [
          "llr",
          "wt_logp",
          "naturalness",
          "pseudo_perplexity",
          "probabilities",
          "predictions",
        ].includes(k)
      )
        continue;
      check(got[k] !== undefined, `Missing runtime trace ${key}`);
      compare(
        key + (index === undefined ? "" : `/${index}`),
        got[k],
        ref(key, index),
      );
    }
  };
  const wt = await engine.run(meta.sequence, {
    mode: "embeddings",
    returnIntermediates: true,
    returnAttentions: true,
    hiddenLayer: 12,
  });
  traces("wt", wt.intermediates);
  compare("wt/llr", wt.llr, ref("wt/llr"));
  compare("selected hidden layer", wt.hiddenState, ref("wt/block.12"));
  const fill = await engine.run(meta.sequence, {
    mode: "fill",
    positions: meta.mask_positions,
    returnIntermediates: true,
  });
  traces("fill", fill.intermediates);
  check(
    fill.residues === meta.sequence &&
      fill.filledSequence.length === meta.sequence.length,
    "Fill residue alignment changed",
  );
  const fillProbs = new Float32Array(meta.mask_positions.length * 20),
    fillPred = new Int32Array(meta.mask_positions.length);
  for (let j = 0; j < meta.mask_positions.length; j++) {
    const p = meta.mask_positions[j];
    fillProbs.set(
      fill.canonicalProbabilities.subarray(p * 20, (p + 1) * 20),
      j * 20,
    );
    fillPred[j] = fill.predictions[p];
  }
  compare("fill/probabilities", fillProbs, ref("fill/probabilities"));
  check(
    fillPred.every((v, i) => v === ref("fill/predictions")[i]),
    "Fill token replay differs",
  );
  const masked = await engine.run(meta.sequence, {
    mode: "masked",
    onTrace: (p, t) => traces("masked", t, p),
    onProgress: (p) => {
      if (p.loaded % 10 === 0) log(`masked ${p.loaded}/${p.total}`);
    },
  });
  compare("masked/llr", masked.llr, ref("masked/llr"));
  compare("masked/wt_logp", masked.wtLogProbs, ref("masked/wt_logp"));
  compare(
    "masked/naturalness",
    [masked.naturalness],
    ref("masked/naturalness"),
  );
  compare(
    "masked/pseudo_perplexity",
    [masked.pseudoPerplexity],
    ref("masked/pseudo_perplexity"),
  );
  // The naturalness route uses the same complete sweep; verify dispatch independently
  // on a small canonical sequence in API tests, and the shared full-sweep outputs here.
  for (let p = 0; p < meta.sequence.length; p++) {
    const j = Array.from(AA_IDS).indexOf(wt.tokens[p + 1]);
    check(masked.llr[p * 20 + j] === 0, "WT diagonal must be exactly zero");
  }
  const times = [];
  if (warm)
    for (let i = 0; i < 3; i++)
      times.push((await engine.forward(meta.sequence)).timingMs);
  return {
    maxRelRms: maxRel,
    tensors,
    coldInferenceMs: wt.timingMs,
    warmMs: times,
    medianMs: times.length ? [...times].sort((a, b) => a - b)[1] : null,
    totalMs: performance.now() - start,
  };
}
export async function verifyTokens(engine, meta, ref) {
  const metrics = {};
  for (const name of ["packed", "padding"]) {
    const out = await engine.forward(ref(name + "/tokens"), {
      sequenceIds: ref(name + "/sequence_id"),
      returnIntermediates: true,
      returnAttentions: true,
    });
    for (const k of Object.keys(meta.tensors).filter((k) =>
      k.startsWith(name + "/"),
    )) {
      const a = out.intermediates[k.slice(name.length + 1)];
      check(a, `Missing ${k}`);
      const s = diffStats(a, ref(k));
      check(s.relRms < 1e-4, `${k}: ${JSON.stringify(s)}`);
      metrics[k] = s;
    }
  }
  return metrics;
}
