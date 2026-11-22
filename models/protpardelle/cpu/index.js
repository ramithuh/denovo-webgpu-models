import { loadBundle, toFloat, abort } from "../shared/loader.js";
import { denoise, validateInput } from "../shared/denoiser.js";
import { graphOps } from "./graph.js";
import { miniMPNN } from "../shared/minimpnn.js";
import { denoiseWithGradient } from "../shared/autodiff.js";
import { sample } from "../shared/sampler.js";
import { constants } from "../shared/constants.js";
import { moments } from "../shared/moments.js";
const f = Math.fround;
const tensor = (data, rows, cols) => ({ data, rows, cols });
export class CPUOps {
  constructor(weights) {
    this.weights = Object.fromEntries(
      Object.entries(weights).map(([k, v]) => [k, { ...v, data: toFloat(v) }]),
    );
  }
  weight(name) {
    const w = this.weights[name];
    if (!w) throw new Error(`Missing weight ${name}`);
    return w;
  }
  linear(x, prefix) {
    const w = this.weight(prefix + ".weight"),
      b = this.weights[prefix + ".bias"];
    const [out, inner] = w.shape;
    if (x.cols !== inner)
      throw new Error(`Linear shape mismatch ${prefix}: ${x.cols}/${inner}`);
    const y = new Float32Array(x.rows * out);
    for (let r = 0; r < x.rows; r++)
      for (let o = 0; o < out; o++) {
        let sum = b?.data[o] ?? 0;
        for (let k = 0; k < inner; k++)
          sum = f(sum + x.data[r * inner + k] * w.data[o * inner + k]);
        y[r * out + o] = sum;
      }
    return tensor(y, x.rows, out);
  }
  norm(x, prefix) {
    const w =
      this.weights[prefix + ".gamma"] ?? this.weight(prefix + ".weight");
    const b = this.weights[prefix + ".beta"] ?? this.weights[prefix + ".bias"];
    const y = new Float32Array(x.data.length),
      d = x.cols;
    for (let r = 0; r < x.rows; r++) {
      const { mean, variance } = moments(x.data, r * d, d);
      const inv = f(1 / f(Math.sqrt(f(variance + f(1e-5)))));
      for (let c = 0; c < d; c++)
        y[r * d + c] =
          f(f(f(x.data[r * d + c] - mean) * inv) * w.data[c]) +
          (b?.data[c] ?? 0);
    }
    return tensor(y, x.rows, d);
  }
  silu(x) {
    return tensor(
      Float32Array.from(x.data, (v) => v / f(1 + f(Math.exp(-v)))),
      x.rows,
      x.cols,
    );
  }
  add(x, y) {
    return tensor(
      Float32Array.from(x.data, (v, i) => v + y.data[i]),
      x.rows,
      x.cols,
    );
  }
  mask(x, mask) {
    return tensor(
      Float32Array.from(x.data, (v, i) => v * mask[Math.floor(i / x.cols)]),
      x.rows,
      x.cols,
    );
  }
  modulate(x, t, mode) {
    return tensor(
      Float32Array.from(x.data, (v, i) => {
        const r = Math.floor(i / x.cols) % t.rows,
          c = i % x.cols,
          offset = r * t.cols;
        return mode === "gate"
          ? v * f(t.data[offset + 2 * x.cols + c] + 1)
          : f(v * f(t.data[offset + c] + 1)) + t.data[offset + x.cols + c];
      }),
      x.rows,
      x.cols,
    );
  }
  noise(input, d) {
    const n = input.seqMask.length,
      values = new Float32Array(n * d);
    for (let r = 0; r < n; r++)
      for (let c = 0; c < d / 2; c++) {
        const freq = constants.noiseFrequencies[d][c],
          time = f(0.25 * f(Math.log(Math.max(1e-6, input.noiseLevel[r]))));
        const angle = f(freq * time);
        values[r * d + c] = Math.cos(angle);
        values[r * d + c + d / 2] = Math.sin(angle);
      }
    return tensor(values, n, d);
  }
  features(input, config) {
    input = {
      ...input,
      coords: input.coords.data ?? input.coords,
      selfCond: input.selfCond?.data ?? input.selfCond,
      cropCond: input.cropCond?.data ?? input.cropCond,
    };
    const m = config.model,
      a = m.struct_model.n_atoms,
      n = input.seqMask.length;
    const concat =
        m.crop_conditional && m.conditioning_style.includes("concat"),
      hotspot = m.crop_conditional && m.conditioning_style.includes("hotspot");
    const channels = 6 + (concat ? 3 : 0) + (hotspot ? 1 : 0),
      sd = config.data.sigma_data;
    const result = new Float32Array(n * a * channels);
    for (let r = 0; r < n; r++) {
      const sigma = Math.max(1e-6, input.noiseLevel[r]),
        scale = f(Math.sqrt(f(f(sigma * sigma) + sd * sd)));
      for (let c = 0; c < channels; c++)
        for (let atom = 0; atom < a; atom++) {
          const j = r * a * 3 + atom * 3;
          result[r * a * channels + c * a + atom] =
            c < 3
              ? input.coords[j + c] / scale
              : c < 6
                ? (input.selfCond?.[j + c - 3] ?? 0)
                : c < 9
                  ? (input.cropCond?.[j + c - 6] ?? 0) / sd
                  : [0, 1, 2, 4].includes(atom)
                    ? (input.hotspotMask?.[r] ?? 0)
                    : 0;
        }
    }
    return tensor(result, n, a * channels);
  }
  attention(q, kv, input, config, prefix) {
    const n = q.rows,
      d = q.cols,
      {
        n_heads: heads,
        dim_head: dh,
        position_embedding_type: position,
        position_embedding_max: max,
      } = config.model.struct_model.uvit;
    const queries = new Float32Array(q.data.length),
      keys = new Float32Array(q.data.length),
      result = new Float32Array(q.data.length),
      probabilities = new Float32Array(heads * n * n);
    const freqs = position.includes("rotary")
      ? this.weight(prefix + "rope.freqs").data
      : null;
    for (let r = 0; r < n; r++)
      for (let h = 0; h < heads; h++)
        for (let c = 0; c < dh; c++) {
          const j = r * d + h * dh + c;
          queries[j] = q.data[j] * f(dh ** -0.5);
          keys[j] = kv.data[r * d * 2 + h * dh + c];
        }
    if (freqs)
      for (let r = 0; r < n; r++)
        for (let h = 0; h < heads; h++)
          for (let c = 0; c < dh; c += 2) {
            const j = r * d + h * dh + c,
              angle = f(
                f(input.residueIndex[r] * freqs[c / 2]) +
                  (input.chainIndex ? f(input.chainIndex[r] * freqs[0]) : 0),
              );
            const cos = f(Math.cos(angle)),
              sin = f(Math.sin(angle));
            for (const values of [queries, keys]) {
              const x = values[j],
                y = values[j + 1];
              values[j] = f(x * cos) - f(y * sin);
              values[j + 1] = f(y * cos) + f(x * sin);
            }
          }
    const rp = this.weights[prefix + "relpos.0.linear.weight"],
      rb = this.weights[prefix + "relpos.0.linear.bias"];
    const cp = this.weights[prefix + "relchain.0.linear.weight"],
      cb = this.weights[prefix + "relchain.0.linear.bias"];
    for (let h = 0; h < heads; h++)
      for (let i = 0; i < n; i++) {
        const scores = new Float32Array(n);
        let largest = -Infinity;
        for (let j = 0; j < n; j++) {
          let score = 0;
          for (let c = 0; c < dh; c++)
            score = f(
              score + queries[i * d + h * dh + c] * keys[j * d + h * dh + c],
            );
          score = f(score);
          let bias = 0;
          if (cp)
            bias = f(
              cp.data[
                h * (max * 2 + 1) +
                  max +
                  +(input.chainIndex[i] !== input.chainIndex[j])
              ] + cb.data[h],
            );
          if (rp) {
            bias = f(
              bias +
                f(
                  rp.data[
                    h * (max * 2 + 1) +
                      max +
                      Math.max(
                        -max,
                        Math.min(
                          max,
                          input.residueIndex[i] - input.residueIndex[j],
                        ),
                      )
                  ] + rb.data[h],
                ),
            );
            if (cp) bias = f(bias * 2);
          }
          score = f(score + bias);
          score = f(score - (1 - input.seqMask[i] * input.seqMask[j]) * 1e6);
          scores[j] = score;
          largest = Math.max(largest, score);
        }
        let sum = 0;
        for (let j = 0; j < n; j++) {
          scores[j] = Math.exp(f(scores[j] - largest));
          sum += scores[j];
        }
        for (let j = 0; j < n; j++) scores[j] /= sum;
        probabilities.set(scores, (h * n + i) * n);
        for (let c = 0; c < dh; c++) {
          let value = 0;
          for (let j = 0; j < n; j++)
            value = f(value + scores[j] * kv.data[j * d * 2 + d + h * dh + c]);
          result[i * d + h * dh + c] = value;
        }
      }
    return {
      ...tensor(result, n, d),
      attentionCache: {
        queries: tensor(queries, n, d),
        keys: tensor(keys, n, d),
        probs: tensor(probabilities, heads * n, n),
        kv,
        input,
        config,
        prefix,
      },
    };
  }
  output(patch, input, config) {
    input = { ...input, coords: input.coords.data ?? input.coords };
    const a = config.model.struct_model.n_atoms,
      sd = config.data.sigma_data,
      result = new Float32Array(patch.data.length);
    for (let r = 0; r < patch.rows; r++) {
      const sigma = Math.max(1e-6, input.noiseLevel[r]),
        variance = f(f(sigma * sigma) + sd * sd);
      const out = f(f(sigma * sd) / f(Math.sqrt(variance))),
        skip = f((sd * sd) / variance);
      for (let atom = 0; atom < a; atom++)
        for (let c = 0; c < 3; c++) {
          const j = r * a * 3 + atom * 3 + c;
          result[j] =
            f(
              f(patch.data[r * a * 3 + c * a + atom] * out) +
                f(input.coords[j] * skip),
            ) * input.seqMask[r];
        }
    }
    return tensor(result, patch.rows, a * 3);
  }
}
Object.assign(CPUOps.prototype, graphOps);
export async function createProtpardelleCPU(options) {
  const { manifest, weights } = await loadBundle(options),
    ops = new CPUOps(weights);
  let destroyed = false;
  return {
    config: manifest.config,
    async sample(
      input,
      { signal, onProgress, returnIntermediates = false } = {},
    ) {
      if (destroyed) throw new Error("Engine destroyed");
      abort(signal);
      if (input.sidechainMode && !ops.weights["mpnn_model.proj_out.weight"]) {
        const base =
          options.miniBaseUrl ??
          options.baseUrl.replace(
            /\/[^/]+\/[^/]+\/?$/,
            "/cc58_epoch97_minimpnn/" + manifest.precision,
          );
        const bundle = await loadBundle({ ...options, baseUrl: base });
        const mini = new CPUOps(bundle.weights);
        for (const [k, v] of Object.entries(mini.weights))
          if (k.startsWith("mpnn_model.")) ops.weights[k] = v;
      }
      const start = performance.now(),
        intermediates = {};
      const result = await sample(ops, manifest.config, input, {
        signal,
        onProgress,
        capture: (name, t) => {
          if (returnIntermediates) intermediates[name] = t.data.slice();
        },
      });
      return {
        coords: result.coords.data,
        aatype: Int32Array.from(result.aatype.data),
        atomMask: result.atomMask.data,
        intermediates,
        timingMs: performance.now() - start,
      };
    },
    gradient(input, { target, lossMask, returnIntermediates = false } = {}) {
      if (destroyed) throw new Error("Engine destroyed");
      validateInput(input, manifest.config);
      const intermediates = {},
        start = performance.now();
      const result = denoiseWithGradient(
        ops,
        manifest.config,
        input,
        target,
        lossMask,
        (name, t) => {
          if (returnIntermediates) intermediates[name] = t.data.slice();
        },
      );
      return {
        coords: result.coords.data,
        gradient: result.gradient.data,
        intermediates,
        timingMs: performance.now() - start,
      };
    },
    sequence(input, { returnIntermediates = false } = {}) {
      if (destroyed) throw new Error("Engine destroyed");
      validateInput(input, manifest.config);
      const start = performance.now(),
        intermediates = {};
      const result = miniMPNN(ops, manifest.config, input, (name, t) => {
        if (returnIntermediates) intermediates[name] = t.data.slice();
      });
      return {
        logprobs: result.data,
        intermediates,
        timingMs: performance.now() - start,
      };
    },
    denoise(input, { signal, returnIntermediates = false } = {}) {
      if (destroyed) throw new Error("Engine destroyed");
      abort(signal);
      validateInput(input, manifest.config);
      const begin = performance.now(),
        intermediates = {};
      const result = denoise(ops, manifest.config, input, (name, value) => {
        if (returnIntermediates) intermediates[name] = value.data.slice();
      });
      abort(signal);
      return {
        coords: result.data,
        intermediates,
        timingMs: performance.now() - begin,
      };
    },
    destroy() {
      if (!destroyed) {
        destroyed = true;
        ops.weights = {};
      }
    },
  };
}
