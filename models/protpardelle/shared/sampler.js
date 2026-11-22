import { denoise } from "./denoiser.js";
import { miniMPNN } from "./minimpnn.js";
import { denoiseWithGradient, kernel } from "./autodiff.js";
import { constants as C } from "./constants.js";
import { abort } from "./loader.js";
import { torchSort21, torchSort21WGSL } from "./sequence.js";
const f = Math.fround,
  L = (x) => Number(x).toExponential(12);
export function noiseSchedule(
  steps,
  { sMin = 0.001, sMax = 80, sigmaData = 10, rho = 7 } = {},
) {
  const result = new Float32Array(steps + 1),
    lo = sMin ** (1 / rho),
    hi = sMax ** (1 / rho);
  result[0] = sigmaData * sMax;
  for (let i = 1; i <= steps; i++) {
    const t = f(1 - i / steps);
    result[i] = f(f(f(hi + f(f(1 - t) * (lo - hi))) ** rho) * sigmaData);
  }
  return result;
}
export class Replay {
  constructor(events) {
    this.events = events;
    this.cursor = 0;
  }
  take(kind, size) {
    if (this.events) {
      const event = this.events[this.cursor++];
      if (!event || event.kind !== kind || event.data.length !== size)
        throw new Error(
          `Replay mismatch at ${this.cursor - 1}: expected ${kind}[${size}], got ${event?.kind}[${event?.data?.length}]`,
        );
      return Float32Array.from(event.data);
    }
    const out = new Float32Array(size);
    if (kind === "numpy_rand" || kind === "categorical_uniform") {
      for (let i = 0; i < size; i++) out[i] = Math.random();
    } else {
      for (let i = 0; i < size; i += 2) {
        const radius = Math.sqrt(-2 * Math.log(Math.max(1e-15, Math.random()))),
          angle = 2 * Math.PI * Math.random();
        out[i] = radius * Math.cos(angle);
        if (i + 1 < size) out[i + 1] = radius * Math.sin(angle);
      }
    }
    return out;
  }
  assertConsumed() {
    if (this.events && this.cursor !== this.events.length)
      throw new Error(
        `Replay has ${this.events.length - this.cursor} unconsumed draws`,
      );
  }
}
function prepareMotif(options, replay, n) {
  if (options.preparedMotif) return options.preparedMotif;
  if (!options.motif) return null;
  const {
    structure,
    sourceIndices,
    targetIndices,
    strategy = "backbone-sidechain",
    translation = [0, 0, 0],
  } = options.motif;
  if (!sourceIndices.length || sourceIndices.length !== targetIndices.length)
    throw new Error("Motif placement shape mismatch");
  if (
    !["backbone-sidechain", "backbone", "sidechain", "sidechain-tip"].includes(
      strategy,
    )
  )
    throw new Error("Unknown crop strategy");
  if (
    sourceIndices.some(
      (i) => !Number.isInteger(i) || i < 0 || i >= structure.aatype.length,
    ) ||
    new Set(targetIndices).size !== targetIndices.length
  )
    throw new Error("Invalid motif indices");
  const raw = replay.take("randn", 4),
    norm = Math.sqrt(raw.reduce((s, x) => f(s + f(x * x)), 0)),
    q = Array.from(raw, (x) => f(x / f(norm))),
    [a, b, c, d] = q;
  const rot = [
    f(f(f(2 * a * a) - 1) + f(2 * b * b)),
    f(f(2 * b * c) - f(2 * a * d)),
    f(f(2 * b * d) + f(2 * a * c)),
    f(f(2 * b * c) + f(2 * a * d)),
    f(f(f(2 * a * a) - 1) + f(2 * c * c)),
    f(f(2 * c * d) - f(2 * a * b)),
    f(f(2 * b * d) - f(2 * a * c)),
    f(f(2 * c * d) + f(2 * a * b)),
    f(f(f(2 * a * a) - 1) + f(2 * d * d)),
  ];
  const center = [0, 0, 0];
  for (const i of sourceIndices)
    for (let c = 0; c < 3; c++)
      center[c] = f(center[c] + structure.coords[i * 111 + 3 + c]);
  for (let c = 0; c < 3; c++) center[c] = f(center[c] / sourceIndices.length);
  const full = new Float32Array(n * 111),
    crop = new Float32Array(n * 111),
    cropSelection = new Float32Array(n * 37),
    lossMask = new Float32Array(n * 37),
    fixed = new Float32Array(n).fill(-1),
    hotspots = new Float32Array(n);
  sourceIndices.forEach((source, index) => {
    const target = targetIndices[index],
      aa = structure.aatype[source];
    if (target < 0 || target >= n)
      throw new Error("Motif index outside sequence");
    fixed[target] = aa;
    hotspots[target] = options.motif.hotspots?.includes(source) ? 1 : 0;
    const tip = C.tips[C.restype1to3[C.restypes[aa]]];
    for (let atom = 0; atom < 37; atom++) {
      const present = structure.atomMask[source * 37 + atom],
        v = [0, 1, 2].map((c) =>
          f(structure.coords[source * 111 + atom * 3 + c] - center[c]),
        );
      const bb = [0, 1, 2, 4].includes(atom);
      const keep =
        strategy === "backbone"
          ? bb
          : strategy === "sidechain"
            ? !bb
            : strategy === "sidechain-tip"
              ? !bb && (!tip || tip.includes(C.atomTypes[atom]))
              : true;
      cropSelection[target * 37 + atom] = +keep;
      for (let c = 0; c < 3; c++) {
        const xyz = f(
          f(f(rot[c * 3] * v[0]) + f(rot[c * 3 + 1] * v[1])) +
            f(rot[c * 3 + 2] * v[2]),
        );
        const value = f(f(xyz * present) + translation[c]);
        full[target * 111 + atom * 3 + c] = value;
        crop[target * 111 + atom * 3 + c] = keep ? value : 0;
      }
      lossMask[target * 37 + atom] = options.tipAtomConditioning
        ? tip?.includes(C.atomTypes[atom])
          ? 1
          : 0
        : 1;
    }
  });
  return { full, crop, cropSelection, lossMask, fixed, hotspots };
}
export async function sample(
  ops,
  config,
  options,
  { signal, onProgress, capture = () => {}, sequenceOps = ops } = {},
) {
  if (
    !options.seqMask?.length ||
    options.residueIndex?.length !== options.seqMask.length ||
    options.chainIndex?.length !== options.seqMask.length
  )
    throw new Error(
      "Sampling requires equal-length seqMask, residueIndex and chainIndex",
    );
  if (
    options.seqMask.some((x) => x !== 0 && x !== 1) ||
    options.residueIndex.some((x) => !Number.isFinite(x)) ||
    options.chainIndex.some((x) => !Number.isInteger(x) || x < 0)
  )
    throw new Error("Invalid sampling masks or indices");
  if (options.useFullMPNN || options.useFullMPNNForFinal)
    throw new Error(
      "External full ProteinMPNN is outside this checkpoint bundle; use MiniMPNN or a separate designer",
    );
  const n = options.seqMask.length,
    mask = options.seqMask,
    numSteps = options.numSteps ?? 200,
    stepScale = options.stepScale ?? 1.2,
    sd = config.data.sigma_data,
    task = config.model.task;
  if (!Number.isInteger(numSteps) || numSteps < 1 || numSteps > 2000)
    throw new Error("numSteps must be 1–2000");
  const tape =
      options.replay instanceof Replay
        ? options.replay
        : new Replay(options.replay),
    schedule =
      options.noiseSchedule ??
      noiseSchedule(numSteps, { ...(options.schedule ?? {}), sigmaData: 10 });
  if (
    schedule.length !== numSteps + 1 ||
    schedule.some((x) => !Number.isFinite(x) || x < 0)
  )
    throw new Error("Invalid noise schedule");
  const pd = options.partial,
    begin = pd ? numSteps - pd.rewindSteps : 0,
    sidechain = !!options.sidechainMode,
    jump = !!options.jumpSteps,
    uniform = !!options.uniformSteps;
  if (
    pd &&
    (!Number.isInteger(pd.rewindSteps) ||
      pd.rewindSteps < 1 ||
      pd.rewindSteps > numSteps)
  )
    throw new Error("Partial rewind must be between 1 and numSteps");
  if (jump && uniform)
    throw new Error("Choose either jumpSteps or uniformSteps");
  if (!["backbone", "ai-allatom", "ai-allatom-nomask"].includes(task))
    throw new Error("Checkpoint does not implement structure sampling");
  const cc = options.conditioning ?? {},
    motif = prepareMotif(options, tape, n),
    reconstruct = !!cc.reconstruction,
    replace = !!cc.replacement;
  if (
    cc.guidanceSchedule &&
    !["constant", "quadratic", "cubic", "custom"].includes(cc.guidanceSchedule)
  )
    throw new Error("Unknown guidance schedule");
  if (!Number.isInteger(cc.recurrence ?? 1) || (cc.recurrence ?? 1) < 1)
    throw new Error("Invalid recurrence count");
  if ((reconstruct || replace || options.motif) && !motif)
    throw new Error("Motif conditioning requires a motif");
  const arr = (a, cols) => ops.fromArray(a, n, cols),
    point = (cols, inputs, code, fn) => kernel(ops, n, cols, inputs, code, fn),
    scale = (x, v) =>
      point(x.cols, [x], `b1[i]=b0[i]*${L(v)};`, (i, r, c, [x]) => x[i] * f(v));
  const maskTensor = arr(mask, 1),
    table37 = ops.fromArray(Float32Array.from(C.atom37Masks.flat()), 21, 37),
    map37 = ops.fromArray(Float32Array.from(C.map37to73.flat()), 21, 37),
    map73 = ops.fromArray(Float32Array.from(C.map73to37.flat()), 21, 73);
  const atomMask = (seq) =>
    point(
      37,
      [seq, table37, maskTensor],
      `b3[i]=b1[u32(b0[r])*37u+c]*b2[r];`,
      (i, r, c, [s, t, m]) => t[s[r] * 37 + c] * m[r],
    );
  const applyMask = (x, m) =>
    point(
      111,
      [x, m],
      `b2[i]=b0[i]*b1[r*37u+c/3u];`,
      (i, r, c, [x, m]) => x[i] * m[r * 37 + Math.floor(c / 3)],
    );
  const fixed = arr(motif?.fixed ?? new Float32Array(n).fill(-1), 1),
    fix = (seq) =>
      point(
        1,
        [seq, fixed],
        `b2[i]=select(b0[i],b1[i],b1[i]>=0.0);`,
        (i, r, c, [s, fixed]) => (fixed[i] >= 0 ? fixed[i] : s[i]),
      );
  const gaussian = (kind = "randn_like", cols = 111) =>
    arr(tape.take(kind, n * cols), cols);
  const sampleSequence = (logprobs) => {
    const uniforms = arr(tape.take("categorical_uniform", n), 1),
      temperature = options.temperature ?? 1,
      topP = options.topP ?? 1,
      omit = options.disallowAAs ?? [4, 20];
    if (temperature <= 0) throw new Error("Temperature must be positive");
    const omitted = ops.fromArray(
      Float32Array.from({ length: 21 }, (_, i) => (omit.includes(i) ? 1 : 0)),
      1,
      21,
    );
    const code = `var probs:array<f32,21>;var order:array<u32,21>;var total=0.0;var largest=-3.402823e38;for(var j=0u;j<21u;j++){largest=max(largest,b0[r*21u+j]);}for(var j=0u;j<21u;j++){probs[j]=exp(b0[r*21u+j]-largest);total+=probs[j];order[j]=j;}for(var j=0u;j<21u;j++){probs[j]/=total;}${torchSort21WGSL}var cumulative=0.0;for(var j=0u;j<21u;j++){cumulative+=probs[order[j]];if(j>0u&&cumulative>${L(topP)}){probs[order[j]]=0.0;}}largest=-3.402823e38;for(var j=0u;j<21u;j++){probs[j]=log(max(probs[j],0.000001))/${L(temperature + 1e-6)}-b2[j]*1000.0;largest=max(largest,probs[j]);}total=0.0;for(var j=0u;j<21u;j++){probs[j]=exp(probs[j]-largest);total+=probs[j];}cumulative=0.0;var choice=20u;for(var j=0u;j<21u;j++){cumulative+=probs[j]/total;if(b1[r]<=cumulative){choice=j;break;}}b3[i]=f32(choice);`;
    return point(
      1,
      [logprobs, uniforms, omitted],
      code,
      (i, r, c, [p, u, omit]) => {
        let vals = Array.from(p.slice(r * 21, (r + 1) * 21)),
          max = Math.max(...vals),
          s = 0;
        vals = vals
          .map((x) => {
            const v = Math.exp(x - max);
            s += v;
            return v;
          })
          .map((x) => x / s);
        const order = torchSort21(vals);
        let cum = 0;
        order.forEach((j, k) => {
          cum += vals[j];
          if (k > 0 && f(cum) > topP) vals[j] = 0;
        });
        vals = vals.map(
          (v, j) =>
            Math.log(Math.max(v, 1e-6)) / (temperature + 1e-6) - omit[j] * 1000,
        );
        max = Math.max(...vals);
        vals = vals.map((v) => Math.exp(v - max));
        s = vals.reduce((a, b) => a + b, 0);
        cum = 0;
        for (let j = 0; j < 21; j++) {
          cum += vals[j] / s;
          if (u[r] <= cum) return j;
        }
        return 20;
      },
    );
  };
  const uniformLogprobs = arr(new Float32Array(n * 21).fill(-Math.log(21)), 21),
    bbSeq = arr(
      Float32Array.from(mask, (x) => x * 7),
      1,
    ),
    bbMask = atomMask(bbSeq);
  const known = options.aatype
    ? arr(options.aatype, 1)
    : pd
      ? arr(pd.structure.aatype, 1)
      : null;
  let seq = known ?? bbSeq,
    sigma = schedule[begin],
    x;
  const fill = (
    x,
    m,
    previous,
    noise,
    level,
    mode = options.dummyFillMode ?? config.data.dummy_fill_mode ?? "zero",
    maskNoise = true,
  ) =>
    point(
      111,
      [x, m, previous ?? x, noise],
      `let mask=b1[r*37u+c/3u];let fill=${mode === "CA" ? "b2[r*111u+3u+c%3u]" : "0.0"};b4[i]=b0[i]*mask+fill*(1.0-mask)+b3[i]*${L(level)}*${maskNoise ? "(1.0-mask)" : "1.0"};`,
      (i, r, c, [x, m, prev, z]) => {
        const mask = m[r * 37 + Math.floor(c / 3)],
          v = mode === "CA" ? prev[r * 111 + 3 + (c % 3)] : 0;
        return (
          f(f(x[i] * mask) + f(v * (1 - mask))) +
          f(f(z[i] * f(level)) * (maskNoise ? 1 - mask : 1))
        );
      },
    );
  if (options.initialCoordinates) x = arr(options.initialCoordinates, 111);
  else if (pd) {
    const coords = arr(pd.structure.coords, 111);
    const center = kernel(
      ops,
      1,
      3,
      [coords],
      `var sum=0.0;for(var j=0u;j<${n}u;j++){sum+=b0[j*111u+3u+c];}b1[i]=sum/${L(n)};`,
      (i, r, c, [x]) => {
        let sum = 0;
        for (let j = 0; j < n; j++) sum = f(sum + x[j * 111 + 3 + c]);
        return sum / n;
      },
    );
    const centered = point(
      111,
      [coords, center],
      `b2[i]=b0[i]-b1[c%3u];`,
      (i, r, c, [x, m]) => x[i] - m[c % 3],
    );
    x = fill(
      centered,
      pd.repack ? bbMask : atomMask(seq),
      null,
      gaussian(),
      sigma,
      options.dummyFillMode ?? config.data.dummy_fill_mode,
      false,
    );
  } else x = scale(gaussian("randn"), schedule[0]);
  x = ops.mask(x, mask);
  if (!jump && !uniform) seq = bbSeq;
  if (!known && (jump || uniform)) seq = sampleSequence(uniformLogprobs);
  let state0 = jump ? ops.zeros(n, 219) : null,
    stateT = jump ? scale(gaussian("randn", 219), sigma) : null,
    lastSigma = jump
      ? arr(new Float32Array(n * 73).fill(sigma), 73)
      : null;
  if (jump && (pd || options.initialCoordinates)) {
    stateT = point(
      219,
      [stateT, x, seq, map73],
      `let mapped=i32(b3[u32(b2[r])*73u+c/3u]);b4[i]=select(b0[i],b1[r*111u+u32(max(mapped,0))*3u+c%3u],mapped>=0);`,
      (i, r, c, [old, xInit, s, map]) => {
        const a = map[s[r] * 73 + Math.floor(c / 3)];
        return a < 0 ? old[i] : xInit[r * 111 + a * 3 + (c % 3)];
      },
    );
  }
  let selfCond = null,
    x0 = null,
    logprobs = null,
    seqSelf = null;
  const motifFull = motif ? arr(motif.full, 111) : null,
    lossMask = motif ? arr(motif.lossMask, 37) : null;
  const crop =
    motif && (cc.cropStart ?? 0) === 0 && cc.crop !== false ? motif.crop : null;
  const outputs = {};
  for (let step = 0; step < numSteps - begin; step++) {
    abort(signal);
    const absStep = begin + step;
    let next = schedule[absStep + 1];
    if (absStep === numSteps - 1) next = 0;
    const gamma =
      next >= (options.sTMin ?? 0.01) * sd && next <= (options.sTMax ?? 50) * sd
        ? Math.min((options.sChurn ?? 50) / numSteps, Math.SQRT2 - 1)
        : 0;
    const runMPNN =
      sidechain &&
      absStep > Math.trunc(numSteps * (options.skipMPNNProportion ?? 0.7));
    let oldSeq = seq,
      m = atomMask(seq);
    if (sidechain && jump) x = fill(x, m, x0, gaussian(), sigma);
    const uniformMask = () => {
      if (uniform) {
        if (known) seq = known;
        else if (logprobs && motif) seq = fix(sampleSequence(logprobs));
        m = atomMask(seq);
      } else if (!jump) m = bbMask;
    };
    uniformMask();
    if (task === "ai-allatom" && uniform) x = applyMask(x, m);
    else if (task === "ai-allatom-nomask" && uniform && x0)
      x = fill(x, m, x0, gaussian(), sigma);
    else if (task === "backbone") x = applyMask(x, bbMask);
    const recurrence = cc.recurrence ?? 1;
    for (let repeat = 0; repeat < recurrence; repeat++) {
      const hat =
        gamma > 0 ? (repeat > 0 ? sigma : f(sigma + f(gamma * sigma))) : sigma;
      const delta =
        repeat > 0
          ? f(Math.sqrt(f(f(sigma * sigma) - f(next * next))))
          : gamma > 0
            ? f(Math.sqrt(f(f(hat * hat) - f(sigma * sigma))))
            : 0;
      let xhat = point(
        111,
        [x, gaussian(), maskTensor],
        `b3[i]=(b0[i]+b1[i]*${L(f(delta * (options.noiseScale ?? 1)))})*b2[r];`,
        (i, r, c, [x, z, m]) =>
          f(x[i] + f(f(z[i] * delta) * (options.noiseScale ?? 1))) * m[r],
      );
      uniformMask();
      if (task === "ai-allatom" && uniform) xhat = applyMask(xhat, m);
      else if (task === "ai-allatom-nomask" && uniform && x0)
        xhat = fill(xhat, m, x0, gaussian(), hat);
      else if (task === "backbone") xhat = applyMask(xhat, bbMask);
      const activeReplace =
        replace &&
        absStep >= (cc.replacementStart ?? 0) * numSteps &&
        absStep < (cc.replacementEnd ?? 0.92) * numSteps;
      let rep = xhat;
      if (activeReplace)
        rep = point(
          111,
          [xhat, motifFull, m, fixed, lossMask],
          `let selected=b3[r]>=0.0&&${options.tipAtomConditioning ? "true" : "b2[r*37u+c/3u]>0.0"}&&b4[r*37u+c/3u]>0.0;b5[i]=select(b0[i],b1[i],selected);`,
          (i, r, c, [x, t, m, fixed, loss]) =>
            fixed[r] >= 0 &&
            (options.tipAtomConditioning || m[r * 37 + Math.floor(c / 3)]) &&
            loss[r * 37 + Math.floor(c / 3)]
              ? t[i]
              : x[i],
        );
      const input = {
        coords: rep,
        noiseLevel: new Float32Array(n).fill(hat),
        seqMask: mask,
        residueIndex: options.residueIndex,
        chainIndex: options.chainIndex,
        hotspotMask: motif?.hotspots,
        selfCond: config.train.self_cond_train_prob > 0.5 ? selfCond : null,
        cropCond: crop,
      };
      const record = (name, value) =>
        capture(`step.${step}.${repeat}.${name}`, value);
      record("input.coords", rep);
      let gradient = null;
      if (reconstruct) {
        const gmask = point(
          37,
          [m, lossMask],
          `b2[i]=b0[i]*b1[i]*${L(cc.lossWeight ?? 1)};`,
          (i, r, c, [a, b]) => a[i] * b[i] * (cc.lossWeight ?? 1),
        );
        const result = denoiseWithGradient(
          ops,
          config,
          input,
          motifFull,
          gmask,
          record,
        );
        x0 = result.coords;
        gradient = result.gradient;
        record("coordinate_gradient", gradient);
        if (activeReplace)
          gradient = point(
            111,
            [gradient, m, fixed, lossMask],
            `let replaced=b2[r]>=0.0&&${options.tipAtomConditioning ? "true" : "b1[r*37u+c/3u]>0.0"}&&b3[r*37u+c/3u]>0.0;b4[i]=select(b0[i],0.0,replaced);`,
            (i, r, c, [g, m, fixed, l]) =>
              fixed[r] >= 0 &&
              (options.tipAtomConditioning || m[r * 37 + Math.floor(c / 3)]) &&
              l[r * 37 + Math.floor(c / 3)]
                ? 0
                : g[i],
          );
      } else x0 = denoise(ops, config, input, record);
      logprobs = runMPNN
        ? ops.mask(
            miniMPNN(
              sequenceOps,
              config,
              {
                ...input,
                coords: x0,
                noiseLevel: new Float32Array(n).fill(hat),
                seqSelfCond: config.model.mpnn_model.use_self_conditioning
                  ? seqSelf
                  : null,
              },
              record,
            ),
            mask,
          )
        : uniformLogprobs;
      selfCond = scale(x0, 1 / sd);
      seqSelf = logprobs;
      if (jump || uniform) {
        let resample = true;
        const anneal = options.annealSequence;
        if (anneal) {
          const skipBegin = Math.trunc(
              numSteps * (options.skipMPNNProportion ?? 0.7),
            ),
            time = 1 - (absStep - skipBegin) / Math.max(1, numSteps - skipBegin),
            rate =
              anneal === "linear"
                ? time
                : (1 + Math.cos(2 * Math.PI * (time - 0.5))) / 2;
          resample = tape.take("numpy_rand", 1)[0] < rate;
        }
        if (!known && resample) seq = sampleSequence(logprobs);
        if (motif && !known) seq = fix(seq);
      }
      let guidanceScale = 0;
      if (
        gradient &&
        absStep >= (cc.reconstructionStart ?? 0) * numSteps &&
        absStep < (cc.reconstructionEnd ?? 2) * numSteps
      ) {
        const schedule = cc.guidanceSchedule ?? "custom";
        guidanceScale =
          (cc.maxScale ?? 10) *
          (schedule === "constant"
            ? 1
            : schedule === "quadratic"
              ? (absStep / numSteps) ** 2
              : schedule === "cubic"
                ? (absStep / numSteps) ** 3
                : absStep < 250
                  ? 0.01
                  : absStep < 400
                    ? 0.1
                    : absStep < 450
                      ? 0.5
                      : 1);
      }
      const guidanceMask = options.tipAtomConditioning
        ? point(
            37,
            [m, fixed, lossMask],
            `b3[i]=b0[i]*select(1.0,b2[i],b1[r]>=0.0);`,
            (i, r, c, [m, fixed, loss]) => m[i] * (fixed[r] >= 0 ? loss[i] : 1),
          )
        : m;
      const euler = (start, clean, prev) => {
        const g = gradient ?? ops.zeros(n, 111);
        return point(
          111,
          [start, clean, prev, g, guidanceMask],
          `let sigma=b2[i];let score=select(0.0,(b0[i]-b1[i])/max(sigma,0.000001),sigma>0.0)+b3[i]*b4[r*37u+c/3u]*${L(guidanceScale)};b5[i]=b0[i]+score*${L(stepScale)}*(${L(next)}-sigma);`,
          (i, r, c, [x, x0, s, g, m]) => {
            const score = f(
              s[i] > 0 ? f(x[i] - x0[i]) / Math.max(s[i], 1e-6) : 0,
            );
            return f(
              x[i] +
                f(
                  f(
                    f(
                      score +
                        f(
                          f(g[i] * m[r * 37 + Math.floor(c / 3)]) *
                            guidanceScale,
                        ),
                    ) * stepScale,
                  ) * f(next - s[i]),
                ),
            );
          },
        );
      };
      if (jump) {
        state0 = point(
          219,
          [state0, x0, oldSeq, map73],
          `let atom=c/3u;let mapped=i32(b3[u32(b2[r])*73u+atom]);b4[i]=select(b0[i],b1[r*111u+u32(max(mapped,0))*3u+c%3u],mapped>=0);`,
          (i, r, c, [old, x0, s, map]) => {
            const a = map[s[r] * 73 + Math.floor(c / 3)];
            return a < 0 ? old[i] : x0[r * 111 + a * 3 + (c % 3)];
          },
        );
        const gather = (state) =>
          point(
            111,
            [state, seq, map37],
            `let mapped=i32(b2[u32(b1[r])*37u+c/3u]);b3[i]=select(0.0,b0[r*219u+u32(max(mapped,0))*3u+c%3u],mapped>=0);`,
            (i, r, c, [s, aa, map]) => {
              const a = map[aa[r] * 37 + Math.floor(c / 3)];
              return a < 0 ? 0 : s[r * 219 + a * 3 + (c % 3)];
            },
          );
        const prev = point(
          111,
          [lastSigma, seq, map37],
          `let mapped=i32(b2[u32(b1[r])*37u+c/3u]);b3[i]=select(${L(hat)},b0[r*73u+u32(max(mapped,0))]*${L(1 + gamma)},mapped>=0);`,
          (i, r, c, [s, aa, map]) => {
            const a = map[aa[r] * 37 + Math.floor(c / 3)];
            return a < 0 ? hat : f(s[r * 73 + a] + f(gamma * s[r * 73 + a]));
          },
        );
        x = euler(gather(stateT), gather(state0), prev);
        stateT = point(
          219,
          [stateT, x, seq, map73],
          `let mapped=i32(b3[u32(b2[r])*73u+c/3u]);b4[i]=select(b0[i],b1[r*111u+u32(max(mapped,0))*3u+c%3u],mapped>=0);`,
          (i, r, c, [old, x, s, map]) => {
            const a = map[s[r] * 73 + Math.floor(c / 3)];
            return a < 0 ? old[i] : x[r * 111 + a * 3 + (c % 3)];
          },
        );
        lastSigma = point(
          73,
          [lastSigma, seq, map73],
          `b3[i]=select(b0[i],${L(next)},b2[u32(b1[r])*73u+c]>=0.0);`,
          (i, r, c, [s, aa, map]) => (map[aa[r] * 73 + c] >= 0 ? next : s[i]),
        );
        oldSeq = seq;
      } else x = euler(xhat, x0, arr(new Float32Array(n * 111).fill(hat), 111));
    }
    sigma = next;
    outputs.coords = x;
    outputs.aatype = seq;
    outputs.atomMask = atomMask(seq);
    capture(
      `trajectory.${step}.coords`,
      scale(x, f(sd / f(Math.sqrt(f(f(next * next) + sd * sd))))),
    );
    capture(`trajectory.${step}.sequence`, seq);
    onProgress?.({
      phase: "sampling",
      loaded: step + 1,
      total: numSteps - begin,
    });
    await ops.stepBoundary?.([
      x,
      x0,
      seq,
      selfCond,
      seqSelf,
      state0,
      stateT,
      lastSigma,
      uniformLogprobs,
      bbSeq,
      bbMask,
      known,
      maskTensor,
      table37,
      map37,
      map73,
      fixed,
      motifFull,
      lossMask,
      ...Object.values(outputs),
    ]);
    // Yield at step boundaries so cancellation reaches command preparation.
    await new Promise((resolve) => setTimeout(resolve, 0));
    abort(signal);
  }
  if (options.stage2) {
    // The upstream stage boundary serializes PDB coordinates to three decimals.
    // Keep that rounding on device, including absent-atom zeros.
    const rounded = point(
      111,
      [x, outputs.atomMask],
      `b2[i]=round(b0[i]*1000.0)*0.001*b1[r*37u+c/3u];`,
      (i, r, c, [x, m]) =>
        (Math.round(x[i] * 1000) / 1000) * m[r * 37 + Math.floor(c / 3)],
    );
    // The original second call reuses the rotated motif, then applies dx/dy/dz
    // again. Translate only placed residues and regenerate the crop selection.
    let secondMotif = motif;
    const translation = options.motif?.translation;
    if (motif && translation?.some((value) => value !== 0)) {
      secondMotif = {
        ...motif,
        full: motif.full.slice(),
        crop: motif.crop.slice(),
      };
      for (let r = 0; r < n; r++) {
        if (motif.fixed[r] < 0) continue;
        for (let atom = 0; atom < 37; atom++)
          for (let c = 0; c < 3; c++) {
            const i = r * 111 + atom * 3 + c;
            secondMotif.full[i] = f(motif.full[i] + translation[c]);
            secondMotif.crop[i] = motif.cropSelection[r * 37 + atom]
              ? secondMotif.full[i]
              : 0;
          }
      }
    }
    const second = {
      ...options,
      ...options.stage2,
      stage2: null,
      replay: tape,
      jumpSteps: false,
      uniformSteps: true,
      aatype: seq,
      partial: {
        structure: { coords: rounded, aatype: seq },
        rewindSteps: options.stage2.rewindSteps ?? 50,
      },
      preparedMotif: secondMotif,
    };
    return sample(ops, config, second, {
      signal,
      onProgress,
      capture: (name, value) => capture("stage2." + name, value),
      sequenceOps,
    });
  }
  tape.assertConsumed();
  return outputs;
}
