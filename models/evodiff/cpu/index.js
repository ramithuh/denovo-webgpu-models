import { loadWeights, decodeTensor } from "../shared/weights.js";
import { normalizeTokens } from "../shared/tokenizer.js";
import { generate, validLength } from "../shared/sampling.js";

// erf approximation (absolute error < 1.5e-7); upstream GELU uses erf, not tanh.
export function gelu(x) {
  const z = Math.abs(x) * Math.SQRT1_2,
    t = 1 / (1 + 0.3275911 * z);
  const erf =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) *
      t +
      0.254829592) *
      t *
      Math.exp(-z * z);
  return x * 0.5 * (1 + (x < 0 ? -erf : erf));
}
export async function createEvoDiffCPU({
  baseUrl,
  precision = "int8",
  ...options
}) {
  const bundle = await loadWeights(
      `${baseUrl.replace(/\/$/, "")}/${precision}`,
      { ...options, expectedPrecision: precision },
    ),
    weights = {};
  for (const [name, t] of Object.entries(bundle.weights))
    weights[name] = decodeTensor(t);
  let destroyed = false,
    busy = false;
  const norm = (x, prefix, C, activation = true) => {
    const out = new Float32Array(x.length),
      w = weights[prefix + ".weight"],
      b = weights[prefix + ".bias"];
    for (let p = 0; p < x.length; p += C) {
      let mean = 0,
        variance = 0;
      for (let c = 0; c < C; c++) mean += x[p + c];
      mean /= C;
      for (let c = 0; c < C; c++) variance += (x[p + c] - mean) ** 2;
      const inv = 1 / Math.sqrt(variance / C + 1e-5);
      for (let c = 0; c < C; c++) {
        const v = Math.fround((x[p + c] - mean) * inv * w[c] + b[c]);
        out[p + c] = activation ? gelu(v) : v;
      }
    }
    return out;
  };
  const conv = (x, prefix, L, CI, CO, KW = 1, dilation = 1, mask = null) => {
    const out = new Float32Array(L * CO),
      w = weights[prefix + ".weight"],
      b = weights[prefix + ".bias"];
    for (let p = 0; p < L; p++)
      for (let o = 0; o < CO; o++) {
        let sum = 0;
        const base = o * CI * KW;
        for (let k = 0; k < KW; k++) {
          const pos = p + (k - (KW >> 1)) * dilation;
          if (pos < 0 || pos >= L) continue;
          const multiplier = mask ? mask[pos] : 1;
          if (multiplier === 0) continue;
          let partial = 0;
          for (let i = 0; i < CI; i++)
            partial += x[pos * CI + i] * w[base + i * KW + k];
          sum += partial * multiplier;
        }
        out[p * CO + o] = sum + b[o];
      }
    return out;
  };
  function forward(
    input,
    { returnIntermediates = false, signal, inputMask } = {},
  ) {
    if (destroyed) throw new Error("Engine destroyed");
    const start = performance.now(),
      tokens = normalizeTokens(input),
      L = validLength(tokens.length),
      trace = {};
    signal?.throwIfAborted();
    if (
      inputMask &&
      (inputMask.length !== L ||
        Array.from(inputMask).some((x) => x !== 0 && x !== 1))
    )
      throw new RangeError("inputMask must contain one 0/1 per position");
    const record = (name, x) => {
      if (returnIntermediates) trace[name] = x.slice();
    };
    let x = new Float32Array(L * 8),
      table = weights["embedder.embedder.weight"];
    for (let p = 0; p < L; p++)
      x.set(table.subarray(tokens[p] * 8, tokens[p] * 8 + 8), p * 8);
    record("embedding", x);
    x = conv(x, "embedder.up_embedder.conv", L, 8, 1024);
    record("projection", x);
    for (let block = 0; block < 16; block++) {
      signal?.throwIfAborted();
      const prefix = `embedder.layers.${block}`;
      let y = norm(x, prefix + ".sequence1.0", 1024);
      y = conv(y, prefix + ".sequence1.2.conv", L, 1024, 512);
      y = norm(y, prefix + ".sequence1.3", 512);
      y = conv(
        y,
        prefix + ".conv",
        L,
        512,
        512,
        5,
        2 ** (block % 8),
        inputMask,
      );
      y = norm(y, prefix + ".sequence2.0", 512);
      y = conv(y, prefix + ".sequence2.2.conv", L, 512, 1024);
      for (let i = 0; i < x.length; i++) x[i] += y[i];
      record(`block.${block}`, x);
    }
    x = norm(x, "last_norm", 1024, false);
    record("final_norm", x);
    const logits = conv(x, "decoder.conv", L, 1024, 31);
    record("logits", logits);
    signal?.throwIfAborted();
    return {
      logits,
      tokens,
      timingMs: performance.now() - start,
      ...(returnIntermediates ? { intermediates: trace } : {}),
    };
  }
  return {
    forward(input, options) {
      if (busy) throw new Error("Engine busy");
      return forward(input, options);
    },
    async generate(options) {
      if (busy || destroyed) throw new Error("Engine unavailable");
      busy = true;
      try {
        return await generate(forward, options);
      } finally {
        busy = false;
      }
    },
    memoryUsage() {
      return {
        weightBytes: Object.values(weights).reduce(
          (a, b) => a + b.byteLength,
          0,
        ),
        transientBuffers: 0,
      };
    },
    destroy() {
      destroyed = true;
      for (const k of Object.keys(weights)) delete weights[k];
    },
  };
}
