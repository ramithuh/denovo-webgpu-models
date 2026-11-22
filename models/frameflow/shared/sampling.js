import { normalizeInput } from "./model.js";
import {
  randomFrames,
  geodesic,
  backbone,
  toPDB,
  adjustOxygen,
  linspaceF32,
} from "./geometry.js";
export async function sample(engine, options = {}) {
  const {
    length,
    steps = 100,
    minT = 0.01,
    expRate = 10,
    schedule = "exp",
    selfCondition = true,
    signal,
    onProgress,
    random,
    initial,
    keepTrajectory = false,
  } = options;
  if (!Number.isSafeInteger(steps) || steps < 2)
    throw new RangeError(
      "steps must be an integer >=2 (upstream linspace convention)",
    );
  if (
    !(minT > 0 && minT < 1) ||
    !(expRate > 0) ||
    !["exp", "linear"].includes(schedule)
  )
    throw new RangeError("Invalid integration schedule");
  let x = normalizeInput(initial ?? randomFrames(length, random));
  const motif = options.motif;
  if (options.guidance)
    throw new Error(
      "Gradient motif guidance is broken in the pinned upstream commit (missing sample_wrapped_normal)",
    );
  if (motif) {
    if (
      motif.mask?.length !== x.length ||
      motif.trans?.length !== x.length * 3 ||
      motif.rotmats?.length !== x.length * 9
    )
      throw new RangeError("Invalid motif frames/mask");
    x.diffuse_mask = Float32Array.from(motif.mask, (v) => 1 - v);
    for (let i = 0; i < x.length; i++)
      if (motif.mask[i]) {
        x.trans_t.set(motif.trans.subarray(i * 3, i * 3 + 3), i * 3);
        x.rotmats_t.set(motif.rotmats.subarray(i * 9, i * 9 + 9), i * 9);
      }
    x = normalizeInput(x);
  }
  const ts = linspaceF32(minT, 1, steps);
  const trajectory = [];
  let pred;
  for (let i = 0; i < steps; i++) {
    signal?.throwIfAborted();
    const t = ts[Math.min(i, steps - 2)]; // Preserve pinned upstream final-call time reuse.
    pred = await engine.forward({ ...x, so3_t: t, r3_t: t }, { signal });
    if (keepTrajectory)
      trajectory.push({
        pred_trans: pred.pred_trans.slice(),
        pred_rotmats: pred.pred_rotmats.slice(),
      });
    onProgress?.({ phase: "step", step: i + 1, total: steps });
    signal?.throwIfAborted();
    if (i === steps - 1) break;
    const dt = Math.fround(ts[i + 1] - ts[i]);
    let trans, rot;
    if (engine.integrate) {
      ({ trans_t: trans, rotmats_t: rot } = await engine.integrate(x, pred, {
        dt,
        t,
        rotationScale: Math.fround(
          dt * (schedule === "exp" ? expRate : 1 / Math.fround(1 - t)),
        ),
        signal,
      }));
    } else {
      trans = new Float32Array(x.trans_t.length);
      rot = new Float32Array(x.rotmats_t.length);
      for (let j = 0; j < trans.length; j++)
        trans[j] =
          x.trans_t[j] +
          Math.fround(
            Math.fround(pred.pred_trans[j] - x.trans_t[j]) / Math.fround(1 - t),
          ) *
            dt;
      for (let j = 0; j < x.length; j++)
        rot.set(
          geodesic(
            x.rotmats_t.subarray(9 * j, 9 * j + 9),
            pred.pred_rotmats.subarray(9 * j, 9 * j + 9),
            Math.fround(
              dt * (schedule === "exp" ? expRate : 1 / Math.fround(1 - t)),
            ),
          ),
          9 * j,
        );
    }
    // Frame clamping accepts explicit fixed frames; it is not the separate amortized checkpoint.
    for (let j = 0; j < x.length; j++)
      if (!x.diffuse_mask[j]) {
        trans.set(x.trans_t.subarray(j * 3, j * 3 + 3), j * 3);
        rot.set(x.rotmats_t.subarray(j * 9, j * 9 + 9), j * 9);
      }
    const sc = selfCondition ? pred.pred_trans.slice() : undefined;
    if (sc)
      for (let j = 0; j < x.length; j++)
        if (!x.diffuse_mask[j])
          sc.set(x.trans_t.subarray(j * 3, j * 3 + 3), j * 3);
    x = { ...x, trans_t: trans, rotmats_t: rot, trans_sc: sc };
  }
  const coords = adjustOxygen(
    backbone(pred.pred_trans, pred.pred_rotmats),
    x.res_mask,
  );
  return {
    ...pred,
    coords,
    pdb: toPDB(coords),
    ...(keepTrajectory ? { trajectory } : {}),
  };
}
