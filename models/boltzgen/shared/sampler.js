import { sigmaSchedule } from "./random.js";
export function rotation(o, raw) {
  return o.tidy(() => {
    const norm = o.point("sqrt", o.sum(o.mul(raw, raw), 1, true));
    const sign = o.point("where", o.point("lt", o.select(raw, 1, 0), 0), -1, 1);
    const q = o.div(raw, o.mul(norm, o.unsqueeze(sign, -1))),
      [r, i, j, k] = [0, 1, 2, 3].map((n) => o.select(q, 1, n));
    const two = o.div(2, o.sum(o.mul(q, q), 1));
    const diag = (a, b) =>
      o.sub(1, o.mul(two, o.add(o.mul(a, a), o.mul(b, b))));
    const off = (a, b, c, d, minus) =>
      o.mul(
        two,
        minus
          ? o.sub(o.mul(a, b), o.mul(c, d))
          : o.add(o.mul(a, b), o.mul(c, d)),
      );
    const values = [
      diag(j, k),
      off(i, j, k, r, true),
      off(i, k, j, r, false),
      off(i, j, k, r, false),
      diag(i, k),
      off(j, k, i, r, true),
      off(i, k, j, r, true),
      off(j, k, i, r, false),
      diag(i, j),
    ];
    return o.reshape(o.cat(values.map((v) => o.unsqueeze(v, -1))), [1, 3, 3]);
  });
}
export function center(o, x, mask) {
  return o.sub(
    x,
    o.div(
      o.sum(o.mul(x, o.unsqueeze(mask, -1)), 1, true),
      o.sum(o.unsqueeze(mask, -1), 1, true),
    ),
  );
}
export function rigidAlign(o, x, y, mask) {
  return o.tidy(() => {
    const w = o.unsqueeze(o.mul(mask, mask), -1),
      den = o.sum(w, 1, true),
      xc = o.div(o.sum(o.mul(x, w), 1, true), den),
      yc = o.div(o.sum(o.mul(y, w), 1, true), den),
      xx = o.sub(x, xc),
      yy = o.sub(y, yc);
    const cov = o.mm(o.transpose(o.mul(w, yy)), xx);
    const [u, s, vh] = o.track(
      o.backend.compute(
        "svd",
        [cov],
        [
          [1, 3, 3],
          [1, 3],
          [1, 3, 3],
        ],
      ),
    );
    const det = o.calc("det", [o.mm(u, vh)], [1]);
    const correction = o.cat([o.fill([1, 2], 1), o.reshape(det, [1, 1])]);
    const rot = o.mm(o.mul(u, o.reshape(correction, [1, 1, 3])), vh);
    return o.add(o.mm(xx, o.transpose(rot)), yc);
  });
}
export async function sample(
  model,
  f,
  conditioning,
  sInputs,
  sTrunk,
  random,
  { numSteps, signal, onProgress, stage },
) {
  const o = model.o,
    sigmas = sigmaSchedule(numSteps),
    mask = f.atom_pad_mask,
    denoised = [],
    trajectory = [];
  let x = o.mul(random[0], sigmas[0]);
  trajectory.push(x);
  for (let i = 0; i < numSteps; i++) {
    signal?.throwIfAborted();
    const sigma = sigmas[i],
      next = sigmas[i + 1],
      gamma = next > 1 ? Math.fround(0.8) : 0,
      tHat = sigma * (1 + gamma);
    const noisy = o.tidy(() => {
      let a = center(o, x, mask);
      a = o.add(o.mm(a, rotation(o, random[i * 3 + 1])), random[i * 3 + 2]);
      return o.add(
        a,
        o.mul(
          random[i * 3 + 3],
          Math.sqrt(Math.fround(tHat * tHat - sigma * sigma)),
        ),
      );
    });
    const den = await model.denoise(
      f,
      conditioning,
      sInputs,
      sTrunk,
      noisy,
      Math.fround(tHat),
      { signal, onProgress, step: i },
    );
    denoised.push(den);
    await stage?.("denoised." + i, den);
    x = o.tidy(() => {
      const aligned = rigidAlign(o, noisy, den, mask);
      return o.add(
        aligned,
        o.mul(o.div(o.sub(aligned, den), tHat), Math.fround(next - tHat)),
      );
    });
    trajectory.push(x);
    await stage?.("trajectory." + (i + 1), x);
    o.collect(f, conditioning, sInputs, sTrunk, random, trajectory, denoised);
    await o.backend.flush?.(true);
    onProgress?.({ phase: "diffusion", step: i + 1, total: numSteps });
  }
  trajectory.push(x);
  return { atom_coords: x, trajectory, denoised };
}
