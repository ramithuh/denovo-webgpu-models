import { FoldFlow2Network } from "./network.js";
import {
  canonical,
  vectorfields,
  integrate,
  backbone,
  expandAtoms,
} from "./geometry.js";
import { abort } from "./loader.js";
export function initialFrames(length, seed = 123) {
  let state = seed >>> 0;
  const uniform = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state + 0.5) / 4294967296;
  };
  const normal = () =>
    Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
  const a = new Float32Array(length * 7);
  for (let i = 0; i < length; i++) {
    let q = Array.from({ length: 4 }, normal),
      n = Math.hypot(...q);
    for (let j = 0; j < 4; j++) a[i * 7 + j] = q[j] / n;
    for (let j = 4; j < 7; j++) a[i * 7 + j] = 10 * normal();
  }
  for (let c = 4; c < 7; c++) {
    let sum = 0;
    for (let i = 0; i < length; i++) sum += a[i * 7 + c];
    for (let i = 0; i < length; i++) a[i * 7 + c] -= sum / length;
  }
  return a;
}
export class Engine {
  constructor(o, w, manifest) {
    this.o = o;
    this.network = new FoldFlow2Network(o, w);
    this.manifest = manifest;
    this.destroyed = false;
    this.busy = false;
  }
  memoryUsage() {
    return this.o.memoryUsage();
  }
  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this.controller?.abort();
    this.o.destroy();
    this.network.w = {};
    this.network.trace = {};
    this.network.keep = [];
    if (this.ownsDevice) this.device.destroy();
  }
  async run({
    length,
    initial,
    steps = 0,
    time = 0.6,
    minTime = 0.01,
    selfConditioning = true,
    seed = 123,
    seqEmbed,
    esmTokens,
    seqIndices,
    scCa,
    signal,
    onProgress,
    trace = false,
    rotationScaling = 10,
  } = {}) {
    if (this.destroyed) throw Error("Engine destroyed");
    if (this.busy) throw Error("Engine is busy");
    length ??= initial?.length / 7;
    if (!Number.isInteger(length) || length < 1)
      throw RangeError("length must be a positive integer");
    if (!Number.isInteger(steps) || steps < 0 || steps === 1)
      throw RangeError("steps must be 0 (forward) or at least 2");
    if (!(time > 0 && time <= 1 && minTime > 0 && minTime < 1))
      throw RangeError("time range");
    if (initial && initial.length !== length * 7)
      throw RangeError("initial frame shape");
    this.busy = true;
    this.controller = new AbortController();
    const externalSignal = signal;
    const relay = () => this.controller?.abort(externalSignal.reason);
    externalSignal?.addEventListener("abort", relay, { once: true });
    if (externalSignal?.aborted) relay();
    signal = this.controller.signal;
    const o = this.o,
      n = this.network;
    n.signal = signal;
    n.onProgress = onProgress;
    n.tracing = trace;
    n.trace = {};
    n.keep = [];
    try {
      abort(signal);
      const idx = seqIndices
        ? Array.from(seqIndices)
        : Array.from({ length }, (_, i) => i + 1);
      if (idx.length !== length || idx.some((i) => !Number.isSafeInteger(i)))
        throw RangeError("sequence index shape or value");
      let frames = o.tensor(
        [length, 7],
        initial ?? initialFrames(length, seed),
      );
      n.keep = [frames];
      let raw;
      if (seqEmbed) {
        if (
          seqEmbed.single.length !== length * 34 * 1280 ||
          seqEmbed.pair.length !== length * length * 660
        )
          throw RangeError("ESM representation shape");
        raw = [
          o.tensor([length, 34, 1280], seqEmbed.single),
          o.tensor([length, length, 660], seqEmbed.pair),
        ];
      } else raw = await n.esm(length, esmTokens);
      await n.mark("esm.single", raw[0]);
      await n.mark("esm.pair", raw[1]);
      const seq = n.sequence(...raw, idx);
      await n.mark("sequence.single", seq[0]);
      await n.mark("sequence.pair", seq[1]);
      if (scCa && scCa.length !== length * 3)
        throw RangeError("self-conditioning shape");
      const idxTensor = o.fastEmbedFeatures ? o.tensor([length], idx) : null;
      let sc = scCa ? o.tensor([length, 3], scCa) : o.constant([length, 3]),
        pred,
        lastInput,
        trajectory = [],
        trajTensors = [];
      n.keep = [frames, idxTensor, ...seq, sc];
      if (steps && selfConditioning) {
        pred = await n.forward(frames, 1, sc, seq, idx, idxTensor);
        sc = pred.t;
        n.keep = [frames, idxTensor, ...seq, sc];
        await n.boundary(n.keep);
      }
      for (let step = 0; step < (steps || 1); step++) {
        abort(signal);
        const t = steps ? 1 - (step * (1 - minTime)) / (steps - 1) : time;
        n.keep = [frames, idxTensor, ...seq, sc, ...trajTensors];
        lastInput = frames;
        pred = await n.forward(frames, t, sc, seq, idx, idxTensor);
        if (steps) {
          frames = integrate(
            o,
            frames,
            pred,
            t,
            (1 - minTime) / (steps - 1),
            rotationScaling,
          );
          if (o.readMany) trajTensors.push(frames);
          else trajectory.push(await o.read(frames));
          if (selfConditioning) sc = pred.t;
        }
        onProgress?.({ phase: "inference", step: step + 1, total: steps || 1 });
        abort(signal);
        if (step < (steps || 1) - 1) {
          n.keep = [frames, idxTensor, ...seq, sc, ...trajTensors];
          if (o.stepBoundary) {
            await o.stepBoundary(n.keep, step);
            abort(signal);
          } else {
            await n.boundary(n.keep);
          }
        }
      }
      const vf = vectorfields(
          o,
          lastInput,
          pred,
          steps ? minTime : time,
          rotationScaling,
        ),
        atomTensor = steps
          ? backbone(
              o,
              o.slice(frames, 1, 0, 4),
              o.slice(frames, 1, 4, 7),
              pred.psi,
            )
          : pred.atoms,
        rigidsTensor = o.cat([canonical(o, pred.q), pred.t]),
        atom37Tensor = expandAtoms(o, atomTensor, 37),
        atom14Tensor = expandAtoms(o, atomTensor, 14);
      let result;
      if (o.readMany) {
        const toRead = [
          rigidsTensor,
          pred.psi,
          atomTensor,
          atom37Tensor,
          atom14Tensor,
          vf.rot,
          vf.trans,
          ...(steps ? [frames, ...trajTensors] : []),
        ];
        const arrays = await o.readMany(toRead);
        result = {
          length,
          steps,
          rigids: arrays[0],
          psi: arrays[1],
          atoms: arrays[2],
          atom37: arrays[3],
          atom14: arrays[4],
          rotVectorfield: arrays[5],
          transVectorfield: arrays[6],
          trace: n.trace,
        };
        if (steps) {
          result.frames = arrays[7];
          result.trajectory = arrays.slice(8);
        }
      } else {
        result = {
          length,
          steps,
          rigids: await o.read(rigidsTensor),
          psi: await o.read(pred.psi),
          atoms: await o.read(atomTensor),
          atom37: await o.read(atom37Tensor),
          atom14: await o.read(atom14Tensor),
          rotVectorfield: await o.read(vf.rot),
          transVectorfield: await o.read(vf.trans),
          trace: n.trace,
        };
        if (steps) {
          result.frames = await o.read(frames);
          result.trajectory = trajectory;
        }
      }
      await o.checkpoint([frames]);
      return result;
    } finally {
      n.keep = [];
      n.trace = {};
      externalSignal?.removeEventListener("abort", relay);
      this.controller = null;
      o.clear();
      this.busy = false;
    }
  }
  forward(opts) {
    return this.run({ ...opts, steps: 0 });
  }
  sample(opts) {
    return this.run({ ...opts, steps: opts?.steps ?? 50 });
  }
}
