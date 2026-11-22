import {
  prepareInput,
  AA_IDS,
  decodeTensor,
  loadWeights,
} from "../shared/index.js";
import { runTask } from "../shared/tasks.js";

export function linear(x, w, b, rows, dim, outDim) {
  const y = new Float32Array(rows * outDim);
  for (let r = 0; r < rows; r++)
    for (let o = 0; o < outDim; o++) {
      let s0 = 0,
        s1 = 0,
        s2 = 0,
        s3 = 0,
        k = 0;
      const xi = r * dim,
        wi = o * dim;
      for (; k + 3 < dim; k += 4) {
        s0 += x[xi + k] * w[wi + k];
        s1 += x[xi + k + 1] * w[wi + k + 1];
        s2 += x[xi + k + 2] * w[wi + k + 2];
        s3 += x[xi + k + 3] * w[wi + k + 3];
      }
      let s = s0 + s1 + s2 + s3;
      for (; k < dim; k++) s += x[xi + k] * w[wi + k];
      y[r * outDim + o] = s + (b?.[o] ?? 0);
    }
  return y;
}
export function layerNorm(x, w, b, rows, dim, eps = 1e-5) {
  const y = new Float32Array(x.length);
  for (let r = 0; r < rows; r++) {
    const off = r * dim;
    let mean = 0,
      v = 0;
    for (let j = 0; j < dim; j++) mean += x[off + j];
    mean /= dim;
    for (let j = 0; j < dim; j++) v += (x[off + j] - mean) ** 2;
    const inv = 1 / Math.sqrt(v / dim + eps);
    for (let j = 0; j < dim; j++)
      y[off + j] = (x[off + j] - mean) * inv * w[j] + (b?.[j] ?? 0);
  }
  return y;
}
function rope(x, L, D, H) {
  const dh = D / H,
    half = dh / 2,
    out = new Float32Array(x.length);
  for (let p = 0; p < L; p++)
    for (let h = 0; h < H; h++)
      for (let j = 0; j < half; j++) {
        const a = Math.fround(p * Math.fround(1 / 10000 ** ((2 * j) / dh))),
          c = Math.fround(Math.cos(a)),
          s = Math.fround(Math.sin(a)),
          i = p * D + h * dh + j;
        out[i] = Math.fround(x[i] * c) - Math.fround(x[i + half] * s);
        out[i + half] = Math.fround(x[i + half] * c) + Math.fround(x[i] * s);
      }
  return out;
}
function attention(q, k, v, L, D, H, ids, maps) {
  const dh = D / H,
    y = new Float32Array(L * D),
    p = new Float32Array(L);
  for (let h = 0; h < H; h++)
    for (let i = 0; i < L; i++) {
      let max = -Infinity;
      for (let j = 0; j < L; j++) {
        let s = 0;
        for (let d = 0; d < dh; d++)
          s += q[i * D + h * dh + d] * k[j * D + h * dh + d];
        p[j] = ids[i] === ids[j] ? s / Math.sqrt(dh) : -Infinity;
        max = Math.max(max, p[j]);
      }
      let sum = 0;
      for (let j = 0; j < L; j++) {
        p[j] = Math.exp(p[j] - max);
        sum += p[j];
      }
      for (let j = 0; j < L; j++) p[j] /= sum;
      if (maps) maps.set(p, (h * L + i) * L);
      for (let d = 0; d < dh; d++) {
        let s = 0;
        for (let j = 0; j < L; j++) s += p[j] * v[j * D + h * dh + d];
        y[i * D + h * dh + d] = s;
      }
    }
  return y;
}
function residual(x, y, scale = 1) {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] + Math.fround(y[i] / scale);
  return out;
}
function gelu(z) {
  const x = Math.abs(z) / Math.SQRT2,
    t = 1 / (1 + 0.3275911 * x);
  const erf =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) *
      t +
      0.254829592) *
      t *
      Math.exp(-x * x);
  return 0.5 * z * (1 + (z < 0 ? -erf : erf));
}
export class EsmcCPU {
  constructor(bundle) {
    this.config = bundle.config;
    this.manifest = bundle.manifest;
    this.w = {};
    for (const [k, v] of Object.entries(bundle.weights))
      this.w[k] = decodeTensor(v);
    this.destroyed = false;
  }
  forward(
    input,
    {
      returnIntermediates = false,
      returnAttentions = false,
      hiddenLayer,
      traceRow,
      sequenceIds,
      wildTypeTokens,
      signal,
    } = {},
  ) {
    if (this.destroyed) throw new Error("Engine destroyed");
    signal?.throwIfAborted();
    const start = performance.now(),
      {
        tokens,
        sequenceIds: ids,
        residues,
      } = prepareInput(input, this.config.max_sequence_length, sequenceIds),
      L = tokens.length,
      D = this.config.d_model,
      w = this.w,
      traces = { tokens, sequence_id: ids },
      attentions = [];
    if (
      hiddenLayer !== undefined &&
      (!Number.isInteger(hiddenLayer) ||
        hiddenLayer < 0 ||
        hiddenLayer >= this.config.n_layers)
    )
      throw new RangeError("Invalid hidden layer");
    if (
      traceRow !== undefined &&
      (!Number.isInteger(traceRow) || traceRow < 0 || traceRow >= L)
    )
      throw new RangeError("Invalid trace row");
    if (
      wildTypeTokens !== undefined &&
      (wildTypeTokens.length !== L ||
        Array.from(wildTypeTokens).some(
          (t) => !Number.isInteger(t) || t < 0 || t >= 64,
        ))
    )
      throw new RangeError("Invalid wildTypeTokens");
    const save = (n, v, dim = D) => {
      if (returnIntermediates)
        traces[n] =
          traceRow === undefined
            ? v
            : v.slice(traceRow * dim, (traceRow + 1) * dim);
    };
    const ln = (x, p) => layerNorm(x, w[p + ".weight"], w[p + ".bias"], L, D);
    const lin = (x, p, di = D, do_ = D) =>
      linear(x, w[p + ".weight"], w[p + ".bias"], L, di, do_);
    const split = (qkv) => {
      const q = new Float32Array(L * D),
        k = new Float32Array(L * D),
        v = new Float32Array(L * D);
      for (let i = 0; i < L; i++) {
        q.set(qkv.subarray(i * 3 * D, i * 3 * D + D), i * D);
        k.set(qkv.subarray(i * 3 * D + D, i * 3 * D + 2 * D), i * D);
        v.set(qkv.subarray(i * 3 * D + 2 * D, (i + 1) * 3 * D), i * D);
      }
      return [q, k, v];
    };
    let x = new Float32Array(L * D),
      hiddenState;
    for (let i = 0; i < L; i++)
      x.set(
        w["embed.weight"].subarray(tokens[i] * D, (tokens[i] + 1) * D),
        i * D,
      );
    save("embedding", x);
    for (let i = 0; i < this.config.n_layers; i++) {
      signal?.throwIfAborted();
      const p = `transformer.blocks.${i}`,
        F = this.config.ffn_size,
        H = this.config.n_heads;
      let [q, k, v] = split(
        lin(
          ln(x, p + ".attn.layernorm_qkv.0"),
          p + ".attn.layernorm_qkv.1",
          D,
          3 * D,
        ),
      );
      q = rope(ln(q, p + ".attn.q_ln"), L, D, H);
      k = rope(ln(k, p + ".attn.k_ln"), L, D, H);
      const maps = returnAttentions ? new Float32Array(H * L * L) : null;
      x = residual(
        x,
        lin(attention(q, k, v, L, D, H, ids, maps), p + ".attn.out_proj"),
        this.config.residual_scale,
      );
      if (maps) {
        attentions.push(maps);
        if (returnIntermediates) traces[`attention.${i}`] = maps;
      }
      const a = lin(ln(x, p + ".ffn.0"), p + ".ffn.1", D, 2 * F),
        act = new Float32Array(L * F);
      for (let r = 0; r < L; r++)
        for (let j = 0; j < F; j++) {
          const z = a[r * 2 * F + j];
          act[r * F + j] =
            Math.fround(z / (1 + Math.exp(-z))) * a[r * 2 * F + F + j];
        }
      x = residual(x, lin(act, p + ".ffn.3", F, D), this.config.residual_scale);
      save(`block.${i}`, x);
      if (i === hiddenLayer) hiddenState = x;
    }
    x = ln(x, "transformer.norm");
    save("final_ln", x);
    const tokenEmbeddings = x,
      embeddings = x.slice(D, (L - 1) * D),
      pooled = new Float32Array(D);
    for (let d = 0; d < D; d++) {
      let sum = 0;
      for (let r = 1; r < L - 1; r++) sum += x[r * D + d];
      pooled[d] = sum / (L - 2);
    }
    if (returnIntermediates && traceRow === undefined) {
      traces.residue_embeddings = embeddings;
      traces.pooled = pooled;
    }
    x = lin(x, "sequence_head.0");
    save("head.0", x);
    x = Float32Array.from(x, gelu);
    save("head.1", x);
    x = ln(x, "sequence_head.2");
    save("head.2", x);
    const logits = lin(x, "sequence_head.3", D, 64);
    save("head.3", logits, 64);
    save("logits", logits, 64);
    const llr = new Float32Array((L - 2) * 20),
      wtLogProbs = new Float32Array(L - 2),
      canonicalProbabilities = new Float32Array((L - 2) * 20),
      predictions = new Int32Array(L - 2);
    const wt = wildTypeTokens ?? tokens;
    if (wt.length !== L) throw new RangeError("wildTypeTokens length mismatch");
    for (let r = 1; r < L - 1; r++) {
      let max = -Infinity,
        sum = 0,
        cm = -Infinity,
        cs = 0,
        best = 0;
      for (let j = 0; j < 64; j++) max = Math.max(max, logits[r * 64 + j]);
      for (let j = 0; j < 64; j++) sum += Math.exp(logits[r * 64 + j] - max);
      wtLogProbs[r - 1] = logits[r * 64 + wt[r]] - max - Math.log(sum);
      for (let j = 0; j < 20; j++)
        if (logits[r * 64 + AA_IDS[j]] > cm) {
          cm = logits[r * 64 + AA_IDS[j]];
          best = j;
        }
      for (let j = 0; j < 20; j++)
        cs += Math.exp(logits[r * 64 + AA_IDS[j]] - cm);
      for (let j = 0; j < 20; j++) {
        llr[(r - 1) * 20 + j] =
          logits[r * 64 + AA_IDS[j]] - logits[r * 64 + wt[r]];
        canonicalProbabilities[(r - 1) * 20 + j] =
          Math.exp(logits[r * 64 + AA_IDS[j]] - cm) / cs;
      }
      predictions[r - 1] = AA_IDS[best];
    }
    return {
      tokens,
      residues,
      logits,
      tokenEmbeddings,
      embeddings,
      pooled,
      hiddenState,
      llr,
      wtLogProbs,
      canonicalProbabilities,
      predictions,
      ...(returnAttentions ? { attentions } : {}),
      ...(returnIntermediates ? { intermediates: traces } : {}),
      timingMs: performance.now() - start,
    };
  }
  summarize(logp, { signal } = {}) {
    signal?.throwIfAborted();
    const naturalness = logp.reduce((a, b) => a + b, 0) / logp.length;
    return { naturalness, pseudoPerplexity: Math.exp(-naturalness) };
  }
  run(sequence, options) {
    return runTask(this, sequence, options);
  }
  async batch(sequences, options) {
    const out = [];
    for (const s of sequences) out.push(await this.run(s, options));
    return out;
  }
  destroy() {
    this.w = {};
    this.destroyed = true;
  }
}
export async function createEsmcCPU({
  baseUrl,
  precision = "fp32",
  ...options
} = {}) {
  return new EsmcCPU(
    await loadWeights(`${baseUrl.replace(/\/$/, "")}/${precision}`, options),
  );
}
