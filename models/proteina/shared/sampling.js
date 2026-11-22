const f = Math.fround;
export function schedule(steps, mode = "log", power = 2) {
  if (
    !Number.isInteger(steps) ||
    steps < 1 ||
    !Number.isFinite(power) ||
    power <= 0
  )
    throw RangeError("Invalid schedule");
  const ts = new Float32Array(steps + 1),
    snrToT = (s) => {
      const r = f(Math.sqrt(Math.max(1e-10, Math.min(1e10, s))));
      return f(r / f(1 + r));
    };
  for (let i = 0; i <= steps; i++) {
    const increment = f(1 / steps);
    const t =
      i < Math.floor((steps + 1) / 2)
        ? f(increment * i)
        : f(1 - f(increment * (steps - i)));
    if (mode === "uniform" || mode === "linear") ts[i] = t;
    else if (mode === "power") ts[i] = t ** power;
    else if (mode === "log") ts[i] = f(1 - f(10 ** ((-power * i) / steps)));
    else if (mode === "loglinear")
      ts[i] = snrToT(f(10 ** (-6 + (12 * i) / steps)));
    else if (mode === "edm") {
      const s = f(
        f(80 ** (1 / power)) +
          f(t * f(0.002 ** (1 / power) - 80 ** (1 / power))),
      );
      ts[i] = snrToT(f(1 / f(f(s ** power) ** 2)));
    } else if (mode === "cos_sch_v_snr") {
      const a = f(f(Math.PI * f(1 - t)) / 2),
        num = f(Math.cos(a) + 1e-5),
        den = f(Math.sin(a) + 1e-5);
      ts[i] = snrToT(f(f(num ** power) / f(den ** power)));
    } else throw RangeError("Unknown schedule");
  }
  if (!["uniform", "linear", "power"].includes(mode)) {
    const lo = ts[0],
      hi = f(ts[steps] - lo);
    for (let i = 0; i <= steps; i++) ts[i] = f(f(ts[i] - lo) / hi);
  }
  return ts;
}
export function noiseSchedule(times, mode = "1/t", power = 1, clamp = null) {
  if (
    !["1/t", "us", "tan"].includes(mode) ||
    !Number.isFinite(power) ||
    power <= 0 ||
    (clamp != null && (!Number.isFinite(clamp) || clamp < 0))
  )
    throw RangeError("Invalid g(t)");
  const a = Float32Array.from(times, (t) => {
    t = Math.min(t, f(1 - 1e-5));
    if (mode === "1/t") return f(1 / f(t + 0.01));
    if (mode === "us") return f(f(1 - t) / f(t + 0.01));
    const angle = f(f(f(1 - t) * f(Math.PI)) / 2);
    return f(
      f(f(Math.PI / 2) * f(Math.sin(angle))) / f(f(Math.cos(angle)) + 0.01),
    );
  });
  if (power !== 1) {
    const logs = Float32Array.from(a, Math.log),
      mean = f(logs.reduce((s, x) => s + x, 0) / a.length);
    for (let i = 0; i < a.length; i++) {
      const z = f((1 / (1 + Math.exp(-f(logs[i] - mean)))) ** power),
        c = Math.min(1 - 1e-6, Math.max(1e-6, z));
      a[i] = f(Math.exp(f(f(Math.log(c / (1 - c))) + mean)));
    }
  }
  if (clamp != null)
    for (let i = 0; i < a.length; i++) a[i] = Math.min(clamp, a[i]);
  return a;
}
export function center(x) {
  const y = new Float32Array(x.length),
    L = x.length / 3;
  for (let c = 0; c < 3; c++) {
    let s = 0;
    for (let i = c; i < x.length; i += 3) s += x[i];
    const m = f(s / L);
    for (let i = c; i < x.length; i += 3) y[i] = f(x[i] - m);
  }
  return y;
}
export function normalRandom(seed = 1) {
  let s = seed >>> 0;
  const uniform = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return (n) => {
    const x = new Float32Array(n);
    for (let i = 0; i < n; i += 2) {
      const r = Math.sqrt(-2 * Math.log(Math.max(uniform(), 1 / 4294967296))),
        a = 2 * Math.PI * uniform();
      x[i] = r * Math.cos(a);
      if (i + 1 < n) x[i + 1] = r * Math.sin(a);
    }
    return x;
  };
}
export function cleanPrediction(x, pred, t) {
  return Float32Array.from(x, (v, i) => f(v + f(f(1 - t) * pred[i])));
}
export function flowStep(x, clean, t, dt, gt, mode, noiseScale, noise) {
  const xc = center(x),
    cc = center(clean),
    v = Float32Array.from(xc, (a, i) => f(f(cc[i] - a) / f(1 - t))),
    out = new Float32Array(x.length);
  const sigma = f(Math.sqrt(f(f(f(2 * gt) * noiseScale) * dt)));
  for (let i = 0; i < x.length; i++) {
    if (mode === "vf") out[i] = f(x[i] + f(v[i] * dt));
    else {
      const score = f(f(f(t * v[i]) - x[i]) / f(1 - t));
      out[i] = f(
        x[i] + f(f(f(v[i] + f(gt * score)) * dt) + f(sigma * noise[i])),
      );
    }
  }
  return center(out);
}
export async function sampleWith(forward, options = {}) {
  const {
    length,
    steps = 400,
    schedule: modeSchedule = "log",
    schedulePower = 2,
    mode = "sc",
    noiseScale = 0.45,
    scoreScale = 1,
    gtMode = "1/t",
    gtPower = 1,
    gtClamp = null,
    selfCondition = true,
    seed = 5,
    noise,
    fold,
    guidanceWeight = 1,
    signal,
    onProgress,
    trajectory = false,
  } = options;
  if (
    !Number.isSafeInteger(length) ||
    length < 1 ||
    !["vf", "sc"].includes(mode) ||
    !Number.isFinite(noiseScale) ||
    noiseScale < 0 ||
    !Number.isFinite(scoreScale) ||
    scoreScale < 0
  )
    throw RangeError("Invalid sampler options");
  if (!Number.isFinite(guidanceWeight) || (guidanceWeight !== 1 && !fold))
    throw RangeError("Classifier-free guidance requires fold indices");
  signal?.throwIfAborted();
  const ts = schedule(steps, modeSchedule, schedulePower),
    gt = noiseSchedule(ts.subarray(0, steps), gtMode, gtPower, gtClamp),
    random = normalRandom(seed);
  let draw = 0;
  const nextNoise = () => {
    const n = noise ? noise[draw++] : random(length * 3);
    if (
      !n ||
      n.length !== length * 3 ||
      Array.from(n).some((v) => !Number.isFinite(v))
    )
      throw RangeError("Missing or invalid saved noise draw");
    return n;
  };
  let x = center(nextNoise()),
    clean;
  const frames = [];
  let sampleMode = mode;
  for (let step = 0; step < steps; step++) {
    signal?.throwIfAborted();
    const t = ts[step],
      dt = f(ts[step + 1] - t),
      previousClean = clean;
    const result = await forward(x, {
      t,
      xSC: step > 0 && selfCondition ? clean : undefined,
      fold,
      signal,
    });
    clean = cleanPrediction(x, result.coordinates, t);
    if (guidanceWeight !== 1) {
      const uncond = await forward(x, {
        t,
        xSC: step > 0 && selfCondition ? previousClean : undefined,
        signal,
      });
      const uc = cleanPrediction(x, uncond.coordinates, t);
      clean = Float32Array.from(clean, (v, i) =>
        f(f(guidanceWeight * v) + f(f(1 - guidanceWeight) * uc[i])),
      );
    }
    if (
      t > f(0.99) ||
      (["cos_sch_v_snr", "edm"].includes(modeSchedule) && t > f(0.985))
    )
      sampleMode = "vf";
    x = flowStep(
      x,
      clean,
      t,
      dt,
      gt[step],
      sampleMode,
      noiseScale,
      sampleMode === "sc" ? nextNoise() : null,
    );
    if (trajectory) frames.push(x.slice());
    onProgress?.({
      phase: "sampling",
      step: step + 1,
      steps,
      t,
      coordinates: x.slice(),
    });
    signal?.throwIfAborted();
  }
  return { coordinates: x, length, units: "nm", trajectory: frames, times: ts };
}
