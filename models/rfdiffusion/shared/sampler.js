import { prepareInput, seededRandom, gaussian, atomsOf } from "./preprocess.js";
import { parsePDB, parseContig, writePDB } from "./structure.js";
import { rotationUpdate, rotationExp, reverseStep } from "./diffusion.js";
import { backbone } from "./backbone.js";
import { alignMotif } from "./alignment.js";
import { describeSymmetry, rotationCoefficients } from "./symmetry.mjs";
import chemical from "./chemical-constants.js";
import schedule from "./diffusion-constants.js";
import { abort } from "./loader.js";
export function applySymmetry(o, xyz, group) {
  const info = describeSymmetry(group, new Float32Array([0, 0, 0])),
    L = xyz.shape[0],
    A = xyz.shape[1];
  if (L % info.copies)
    throw Error(`Length must be divisible by ${info.copies}`);
  const unit = (L / info.copies) * A,
    coef = o.tensor([info.copies, 3], rotationCoefficients(info));
  return o.kernel(
    xyz.shape,
    [xyz, coef],
    `const point=Math.floor(i/3),k=i%3,src=(point%${unit})*3,r=Math.floor(point/${unit})*3,c=a1[r],s=a1[r+1],sign=a1[r+2],x=a0[src],y=a0[src+1],z=a0[src+2];o[i]=k===0?rfFma(-s,y,Math.fround(c*x)):k===1?sign*rfFma(c,y,Math.fround(s*x)):sign*z;`,
    `let point=i/3u;let k=i%3u;let src=(point%${unit}u)*3u;let r=(point/${unit}u)*3u;let c=a1[r];let s=a1[r+1u];let sign=a1[r+2u];let x=a0[src];let y=a0[src+1u];let z=a0[src+2u];var v=sign*z;if(k==0u){v=fma(-s,y,c*x);}else if(k==1u){v=sign*fma(c,y,s*x);}o[i]=v;`,
  );
}
export async function sample(
  o,
  infer,
  {
    length,
    contig,
    inputPdb,
    symmetry,
    numSteps = 50,
    seed = 0,
    recordedNoise,
    signal,
    onProgress,
    onInputs,
  } = {},
) {
  const start = performance.now(),
    random = seededRandom(seed),
    pdb = inputPdb ? parsePDB(inputPdb) : null,
    mapping = contig ? parseContig(contig, { pdb, random }) : null,
    L = mapping?.length ?? length;
  if (!Number.isInteger(L) || L < 2)
    throw Error("length must be an integer >= 2");
  if (!Number.isInteger(numSteps) || numSteps < 1 || numSteps > 50)
    throw Error("numSteps must be in [1,50]");
  const ids = recordedNoise?.sequence
      ? Int32Array.from(recordedNoise.sequence)
      : Int32Array.from(
          { length: L },
          (_, i) => mapping?.[i]?.source?.sequence ?? 21,
        ),
    mask = recordedNoise?.diffusionMask
      ? Uint8Array.from(recordedNoise.diffusionMask)
      : Uint8Array.from({ length: L }, (_, i) =>
          Number(Boolean(mapping?.[i]?.source)),
        ),
    idx = recordedNoise?.idx
      ? Int32Array.from(recordedNoise.idx)
      : Int32Array.from({ length: L }, (_, i) => mapping?.[i]?.idx ?? i);
  let chains = Array.from({ length: L }, (_, i) =>
    String.fromCharCode(65 + (mapping?.[i]?.chain ?? 0)),
  );
  if (symmetry) {
    const info = describeSymmetry(symmetry, new Float32Array([0, 0, 0]));
    if (L % info.copies)
      throw Error("Length is not divisible by symmetry order");
    const unit = L / info.copies;
    for (let i = 0; i < L; i++) {
      idx[i] += L * (1 + Math.floor(i / unit));
      chains[i] = String.fromCharCode(65 + Math.floor(i / unit));
      ids[i] = ids[i % unit];
      mask[i] = mask[i % unit];
    }
  }
  let coords;
  if (recordedNoise?.initialCoords) {
    coords = Float32Array.from(recordedNoise.initialCoords);
    if (
      coords.length !== L * 9 &&
      coords.length !== L * 12 &&
      coords.length !== L * 42
    )
      throw Error("initialCoords shape mismatch");
  } else {
    coords = new Float32Array(L * 12);
    const rotations = [];
    for (let i = 0; i < L; i++) {
      let axis = [gaussian(random), gaussian(random), gaussian(random)],
        norm = Math.hypot(...axis),
        u = random(),
        hi = schedule.cdf.findIndex((x) => x >= u);
      if (hi < 0) hi = 999;
      const lo = Math.max(0, hi - 1),
        angle =
          lo === hi
            ? schedule.omega[hi]
            : schedule.omega[lo] +
              ((schedule.omega[hi] - schedule.omega[lo]) *
                (u - schedule.cdf[lo])) /
                (schedule.cdf[hi] - schedule.cdf[lo]);
      rotations.push(...axis.map((x) => (x / norm) * angle));
      for (let a = 0; a < 4; a++)
        coords.set(chemical.initial[a], i * 12 + a * 3);
    }
    let x = o.tensor([L, 4, 3], coords),
      R = rotationExp(o, o.tensor([L, 3], rotations)),
      translation = o.tensor(
        [L, 1, 3],
        Array.from(
          { length: L * 3 },
          () =>
            (gaussian(random) * Math.sqrt(1 - schedule.alphabar[49])) / 0.25,
        ),
      );
    x = o.add(o.einsum("nij,naj->nai", R, x), translation);
    coords = await o.read(x);
    const motifCom = [0, 0, 0];
    let motifCount = 0;
    for (let i = 0; i < L; i++) {
      const srcCoords = (mapping?.[i] ?? mapping?.[i % (mapping?.length || L)])?.source?.coords;
      if (mask[i] && srcCoords) {
        motifCom[0] += srcCoords[3];
        motifCom[1] += srcCoords[4];
        motifCom[2] += srcCoords[5];
        motifCount++;
      }
    }
    if (motifCount > 0) {
      motifCom[0] /= motifCount;
      motifCom[1] /= motifCount;
      motifCom[2] /= motifCount;
    }
    const expanded = new Float32Array(L * 42).fill(NaN);
    for (let i = 0; i < L; i++) {
      expanded.set(coords.slice(i * 12, i * 12 + 12), i * 42);
      const srcCoords = (mapping?.[i] ?? mapping?.[i % (mapping?.length || L)])?.source?.coords;
      if (mask[i] && srcCoords) {
        const off = i * 42;
        for (let k = 0; k < 42; k++) {
          const v = srcCoords[k];
          expanded[off + k] = Number.isFinite(v) ? v - motifCom[k % 3] : v;
        }
      }
    }
    coords = expanded;
  }
  if (symmetry && !recordedNoise?.initialCoords)
    coords = await o.read(
      applySymmetry(
        o,
        o.tensor([L, coords.length / L / 3, 3], coords),
        symmetry,
      ),
    );
  o.clear();
  let selfCondition = null,
    trajectory = [],
    last;
  const tSchedule = Array.from({ length: numSteps }, (_, s) =>
    numSteps === 1
      ? 50
      : Math.max(1, Math.min(50, Math.round(50 - (s * 49) / (numSteps - 1)))),
  );
  for (let step = 0; step < numSteps; step++) {
    abort(signal);
    const t = tSchedule[step],
      nextT = step + 1 < numSteps ? tSchedule[step + 1] : 0,
      input = prepareInput({
        length: L,
        coords,
        sequence: ids,
        idx,
        t,
        selfCondition,
        diffusionMask: mask,
      });
    onInputs?.(step, input);
    const predicted = await infer(input, {
      signal,
      onProgress: (p) =>
        onProgress?.({
          ...p,
          blockTotal: p?.blockTotal ?? p?.total,
          step: step + 1,
          numSteps,
          total: numSteps,
          t,
        }),
    });
    let raw = o.tensor([L, 3, 3], predicted.px0),
      px0 = o.tensor([L, 4, 3], predicted.coords);
    if (symmetry) {
      const sym = applySymmetry(o, px0, symmetry);
      const conditioned = o.slice(sym, 1, 0, 3);
      selfCondition = await o.read(conditioned);
      px0 = backbone(
        o,
        conditioned,
        o.tensor([L, 10, 2], predicted.alpha),
        Array.from(ids),
      );
    } else selfCondition = predicted.px0;
    const current = o.tensor([L, coords.length / L / 3, 3], coords);
    let next,
      rotation = null;
    const supplied = recordedNoise?.steps?.[step];
    if (step < numSteps - 1 && t > 1) {
      px0 = alignMotif(o, px0, current, mask);
      const trans = o.tensor(
          [L, 3],
          supplied?.translation ??
            Array.from({ length: L * 3 }, () => gaussian(random)),
        ),
        rot = o.tensor(
          [L, 3],
          supplied?.rotation ??
            Array.from({ length: L * 3 }, () => gaussian(random)),
        ),
        result = reverseStep(o, {
          xt: o.slice(current, 1, 0, 3),
          px0,
          t,
          nextT,
          translationNoise: trans,
          rotationNoise: rot,
          mask,
        });
      next = result.coords;
      rotation = await o.read(result.rotation);
    } else next = px0;
    if (symmetry) next = applySymmetry(o, next, symmetry);
    const px0Array = await o.read(px0);
    const nextCoords = await o.read(next);
    const oldAtoms = atomsOf(coords, L);
    if (next.shape[1] === 3) {
      coords = new Float32Array(L * 42).fill(NaN);
      for (let i = 0; i < L; i++) {
        coords.set(nextCoords.slice(i * 9, i * 9 + 9), i * 42);
        if (mask[i]) coords.set(oldAtoms[i].slice(0, 14).flat(), i * 42);
      }
    } else coords = nextCoords;
    trajectory.push({
      t,
      px0: px0Array,
      xyz: predicted.px0,
      coords: nextCoords,
      rotation,
    });
    last = px0Array;
    o.clear();
    onProgress?.({
      phase: "step",
      step: step + 1,
      numSteps,
      total: numSteps,
      t,
      fraction: (step + 1) / numSteps,
    });
  }
  if (symmetry)
    last = await o.read(applySymmetry(o, o.tensor([L, 4, 3], last), symmetry));
  o.clear();
  const motifIndices = [],
    nativeMotifCoords = [];
  for (let i = 0; i < L; i++) {
    if (mask[i]) {
      const srcCoords = (mapping?.[i] ?? mapping?.[i % (mapping?.length || L)])?.source?.coords;
      if (srcCoords && Number.isFinite(srcCoords[3]) && Number.isFinite(srcCoords[4]) && Number.isFinite(srcCoords[5])) {
        motifIndices.push(i);
        nativeMotifCoords.push([srcCoords[3], srcCoords[4], srcCoords[5]]);
      }
    }
  }
  return {
    pdbText: writePDB(last, ids, { chains }),
    coords: last,
    sequence: ids,
    diffusionMask: mask,
    motifIndices,
    nativeMotifCoords,
    trajectory,
    timingMs: performance.now() - start,
  };
}
