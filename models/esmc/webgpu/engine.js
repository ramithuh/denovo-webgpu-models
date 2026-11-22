import { loadWeights, prepareInput, AA_IDS } from "../shared/index.js";
import { runTask } from "../shared/tasks.js";
import {
  matmulSource,
  normSource,
  elementSource,
  gatherSource,
  ropeSource,
  streamingContextSource,
  scoresSource,
  softmaxSource,
  contextSource,
  geluSource,
  poolSource,
  taskScoresSource,
  summarySource,
} from "./shaders.js";

export async function requestWebGpuDevice({ useF16 = true } = {}) {
  if (!globalThis.navigator?.gpu)
    throw new Error(
      "WebGPU requires a supported browser and HTTPS or localhost",
    );
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) throw new Error("No WebGPU adapter");
  const info = adapter.info ?? (await adapter.requestAdapterInfo?.());
  if (
    adapter.isFallbackAdapter ||
    info?.isFallbackAdapter ||
    /swiftshader|llvmpipe|lavapipe|software/i.test(JSON.stringify(info))
  )
    throw new Error("Hardware WebGPU is required; software adapter rejected");
  const hasF16 = adapter.features.has("shader-f16");
  const device = await adapter.requestDevice({
    requiredFeatures: useF16 && hasF16 ? ["shader-f16"] : [],
  });
  return { adapter, device, hasF16, info };
}
class Engine {
  constructor(device, { ownsDevice = false, useF16 = true } = {}) {
    this.device = device;
    this.ownsDevice = ownsDevice;
    this.useF16 = useF16 && device.features.has("shader-f16");
    this.weights = {};
    this.buffers = new Set();
    this.pipelines = new Map();
    this.destroyed = false;
    this.busy = false;
    this.lost = null;
    device.lost.then((info) => {
      this.lost = info.message || "Device lost";
    });
    this.zero = this.buffer(new Float32Array(4));
  }
  buffer(dataOrSize, temporary = false) {
    const data = typeof dataOrSize === "number" ? null : dataOrSize;
    const size = Math.max(
      4,
      Math.ceil((data?.byteLength ?? dataOrSize) / 4) * 4,
    );
    if (this.destroyed) throw new Error("Engine destroyed");
    if (
      size > this.device.limits.maxBufferSize ||
      size > this.device.limits.maxStorageBufferBindingSize
    )
      throw new RangeError(
        `GPU buffer ${size} bytes exceeds maxBufferSize/maxStorageBufferBindingSize`,
      );
    const buffer =
      (temporary && !data ? this.pool.get(size)?.pop() : null) ??
      this.device.createBuffer({
        size,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_DST |
          GPUBufferUsage.COPY_SRC,
      });
    if (data) {
      const bytes = new Uint8Array(size);
      bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      this.device.queue.writeBuffer(buffer, 0, bytes);
    }
    this.buffers.add(buffer);
    if (temporary) {
      this.temporary.add(buffer);
      this.liveTemps.add(buffer);
    }
    return buffer;
  }
  releaseExcept(current, traces) {
    const keep = new Set([
      current,
      ...Object.values(traces).map((t) => t.buffer),
    ]);
    for (const b of this.liveTemps)
      if (!keep.has(b)) {
        if (!this.pool.has(b.size)) this.pool.set(b.size, []);
        this.pool.get(b.size).push(b);
        this.liveTemps.delete(b);
      }
  }
  memoryUsage() {
    return {
      bufferCount: this.buffers.size,
      transientBuffers: [...(this.temporary ?? [])].filter((b) =>
        this.buffers.has(b),
      ).length,
      totalBytes: [...this.buffers].reduce((n, b) => n + b.size, 0),
    };
  }
  async upload(name, t) {
    if (this.destroyed) throw new Error("Engine destroyed");
    this.weights[name] = {
      buffer: this.buffer(t.bytes),
      dtype: t.dtype,
      shape: t.shape,
      scales: t.scales ? this.buffer(t.scales) : this.zero,
    };
  }
  async pipeline(code, count) {
    let promise = this.pipelines.get(code);
    if (!promise) {
      promise = (async () => {
        const module = this.device.createShaderModule({ code });
        const info = await module.getCompilationInfo();
        const errors = info.messages.filter((m) => m.type === "error");
        if (errors.length)
          throw new Error(errors.map((e) => e.message).join("\n"));
        const layout = this.device.createBindGroupLayout({
          entries: Array.from({ length: count }, (_, binding) => ({
            binding,
            visibility: GPUShaderStage.COMPUTE,
            buffer: {
              type: binding === count - 1 ? "storage" : "read-only-storage",
            },
          })),
        });
        const pipeline = await this.device.createComputePipelineAsync({
          layout: this.device.createPipelineLayout({
            bindGroupLayouts: [layout],
          }),
          compute: { module, entryPoint: "main" },
        });
        return { pipeline, layout };
      })();
      this.pipelines.set(code, promise);
    }
    return promise;
  }
  async dispatch(code, inputs, n, groups) {
    const y = this.buffer(n * 4, true),
      buffers = [...inputs, y];
    const { pipeline, layout } = await this.pipeline(code, buffers.length);
    const bind = this.device.createBindGroup({
      layout,
      entries: buffers.map((buffer, binding) => ({
        binding,
        resource: { buffer },
      })),
    });
    const pass = this.encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(...groups);
    pass.end();
    return y;
  }
  begin() {
    if (this.destroyed) throw new Error("Engine destroyed");
    if (this.lost) throw new Error(this.lost);
    if (this.busy) throw new Error("An inference is already running");
    this.busy = true;
    this.temporary = new Set();
    this.liveTemps = new Set();
    this.pool = new Map();
    this.encoder = this.device.createCommandEncoder();
    this.device.pushErrorScope("validation");
  }
  async end() {
    const validation = await this.device.popErrorScope();
    for (const b of this.temporary) {
      b.destroy();
      this.buffers.delete(b);
    }
    this.temporary.clear();
    this.liveTemps.clear();
    this.pool.clear();
    this.busy = false;
    if (validation) throw new Error(`WebGPU validation: ${validation.message}`);
  }
  async readOutputs(traces, signal) {
    const size = Object.values(traces).reduce((a, t) => a + t.n * 4, 0);
    if (size > this.device.limits.maxBufferSize)
      throw new RangeError(
        "Trace readback exceeds maxBufferSize; reduce sequence length or traces",
      );
    const readback = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.buffers.add(readback);
    this.temporary.add(readback);
    let offset = 0;
    for (const t of Object.values(traces)) {
      t.offset = offset;
      this.encoder.copyBufferToBuffer(t.buffer, 0, readback, offset, t.n * 4);
      offset += t.n * 4;
    }
    signal?.throwIfAborted();
    this.device.queue.submit([this.encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    try {
      const mapped = readback.getMappedRange(),
        values = {};
      for (const [name, t] of Object.entries(traces))
        values[name] = new Float32Array(
          mapped.slice(t.offset, t.offset + t.n * 4),
        );
      signal?.throwIfAborted();
      for (const a of Object.values(values))
        for (const v of a)
          if (!Number.isFinite(v)) throw new Error("Nonfinite GPU output");
      return values;
    } finally {
      readback.unmap();
    }
  }
  async forward(
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
    signal?.throwIfAborted();
    const {
        tokens,
        sequenceIds: ids,
        residues,
      } = prepareInput(input, this.config.max_sequence_length, sequenceIds),
      L = tokens.length,
      D = this.config.d_model;
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
    this.begin();
    const started = performance.now(),
      traces = {},
      w = this.weights;
    const save = (name, value, n = L * D, dim = D) => {
      if (!returnIntermediates) return;
      if (traceRow !== undefined) {
        const b = this.buffer(dim * 4, true);
        this.encoder.copyBufferToBuffer(
          value,
          traceRow * dim * 4,
          b,
          0,
          dim * 4,
        );
        traces[name] = { buffer: b, n: dim };
      } else traces[name] = { buffer: value, n };
    };
    const elem = (n, exp, inputs) =>
      this.dispatch(elementSource(n, exp, inputs.length), inputs, n, [
        Math.ceil(n / 256),
      ]);
    const ln = (x, p, { stride = D, offset = 0 } = {}) =>
      this.dispatch(
        normSource(L, D, { stride, offset, bias: !!w[p + ".bias"] }),
        [x, w[p + ".weight"].buffer, w[p + ".bias"]?.buffer ?? this.zero],
        L * D,
        [L],
      );
    const linear = (x, p, K = D, N = D) => {
      const weight = w[p + ".weight"],
        bias = w[p + ".bias"];
      return this.dispatch(
        matmulSource(L, N, K, weight.dtype, !!bias, this.useF16),
        [x, weight.buffer, bias?.buffer ?? this.zero, weight.scales],
        L * N,
        [Math.ceil(N / 16), Math.ceil(L / 16)],
      );
    };
    const residual = (x, y) =>
      elem(L * D, `y[i]=x[i]+b[i]/${this.config.residual_scale};`, [x, y]);
    try {
      const idBuffer = this.buffer(ids, true),
        tokenBuffer = this.buffer(tokens, true);
      // Input buffers must survive scratch recycling throughout the stack.
      const keep = {
        ids: { buffer: idBuffer },
        tokens: { buffer: tokenBuffer },
      };
      let x = await this.dispatch(
        gatherSource(L, D),
        [tokenBuffer, w["embed.weight"].buffer],
        L * D,
        [Math.ceil((L * D) / 256)],
      );
      save("embedding", x);
      for (let i = 0; i < this.config.n_layers; i++) {
        signal?.throwIfAborted();
        if (this.destroyed) throw new Error("Engine destroyed");
        const p = `transformer.blocks.${i}`,
          F = this.config.ffn_size,
          H = this.config.n_heads;
        const qkv = await linear(
          await ln(x, p + ".attn.layernorm_qkv.0"),
          p + ".attn.layernorm_qkv.1",
          D,
          3 * D,
        );
        let q = await ln(qkv, p + ".attn.q_ln", { stride: 3 * D }),
          k = await ln(qkv, p + ".attn.k_ln", { stride: 3 * D, offset: D });
        q = await this.dispatch(ropeSource(L, D, H), [q], L * D, [
          Math.ceil((L * D) / 256),
        ]);
        k = await this.dispatch(ropeSource(L, D, H), [k], L * D, [
          Math.ceil((L * D) / 256),
        ]);
        let context;
        // Use the register kernel for complete query tiles and their final partial tile.
        // Explicit attention maps retain the materialized probabilities for every layer.
        if (!returnAttentions && L >= 64 && D / H === 64) {
          context = await this.dispatch(
            streamingContextSource(L, H),
            [q, k, qkv, idBuffer],
            L * D,
            [Math.ceil(L / 64), H],
          );
        } else {
          const scores = await this.dispatch(
            scoresSource(L, D, H),
            [q, k, idBuffer],
            H * L * L,
            [
              Math.min(65535, Math.ceil((H * L * L) / 64)),
              Math.ceil((H * L * L) / 64 / 65535),
            ],
          );
          const prob = await this.dispatch(
            softmaxSource(L, H),
            [scores],
            H * L * L,
            [H * L],
          );
          if (returnAttentions)
            traces[`attention.${i}`] = { buffer: prob, n: H * L * L };
          context = await this.dispatch(
            contextSource(L, D, H, { stride: 3 * D, offset: 2 * D }),
            [prob, qkv],
            L * D,
            [Math.ceil((L * D) / 64)],
          );
        }
        x = await residual(x, await linear(context, p + ".attn.out_proj"));
        const ff = await linear(
          await ln(x, p + ".ffn.0"),
          p + ".ffn.1",
          D,
          2 * F,
        );
        const act = await elem(
          L * F,
          `let r=i/${F}u;let j=i%${F}u;let a=x[r*${2 * F}u+j];y[i]=(a/(1.0+exp(-a)))*x[r*${2 * F}u+${F}u+j];`,
          [ff],
        );
        x = await residual(x, await linear(act, p + ".ffn.3", F, D));
        save(`block.${i}`, x);
        if (i === hiddenLayer) traces.hiddenState = { buffer: x, n: L * D };
        this.releaseExcept(x, { ...traces, ...keep });
      }
      x = await ln(x, "transformer.norm");
      save("final_ln", x);
      traces.tokenEmbeddings = { buffer: x, n: L * D };
      traces.pooled = {
        buffer: await this.dispatch(poolSource(L, D), [x], D, [
          Math.ceil(D / 256),
        ]),
        n: D,
      };
      x = await linear(x, "sequence_head.0");
      save("head.0", x);
      x = await this.dispatch(geluSource(L * D), [x], L * D, [
        Math.ceil((L * D) / 256),
      ]);
      save("head.1", x);
      x = await ln(x, "sequence_head.2");
      save("head.2", x);
      const logits = await linear(x, "sequence_head.3", D, 64);
      save("head.3", logits, L * 64, 64);
      save("logits", logits, L * 64, 64);
      traces.outputLogits = { buffer: logits, n: L * 64 };
      const wt = wildTypeTokens
        ? this.buffer(Int32Array.from(wildTypeTokens), true)
        : tokenBuffer;
      traces.taskScores = {
        buffer: await this.dispatch(
          taskScoresSource(L, AA_IDS),
          [logits, wt],
          (L - 2) * 42,
          [Math.ceil((L - 2) / 64)],
        ),
        n: (L - 2) * 42,
      };
      const values = await this.readOutputs(traces, signal),
        embeddings = values.tokenEmbeddings.slice(D, (L - 1) * D);
      const llr = new Float32Array((L - 2) * 20),
        wtLogProbs = new Float32Array(L - 2),
        canonicalProbabilities = new Float32Array((L - 2) * 20),
        predictions = new Int32Array(L - 2);
      for (let p = 0; p < L - 2; p++) {
        llr.set(values.taskScores.subarray(p * 42, p * 42 + 20), p * 20);
        canonicalProbabilities.set(
          values.taskScores.subarray(p * 42 + 20, p * 42 + 40),
          p * 20,
        );
        wtLogProbs[p] = values.taskScores[p * 42 + 40];
        predictions[p] = values.taskScores[p * 42 + 41];
      }
      const out = {
        tokens,
        residues,
        logits: values.outputLogits,
        tokenEmbeddings: values.tokenEmbeddings,
        embeddings,
        pooled: values.pooled,
        hiddenState: values.hiddenState,
        llr,
        wtLogProbs,
        canonicalProbabilities,
        predictions,
        timingMs: performance.now() - started,
      };
      if (returnAttentions)
        out.attentions = Array.from(
          { length: this.config.n_layers },
          (_, i) => values[`attention.${i}`],
        );
      if (returnIntermediates) {
        const {
          outputLogits,
          tokenEmbeddings,
          taskScores,
          hiddenState,
          ...intermediates
        } = values;
        if (traceRow === undefined)
          intermediates.residue_embeddings = embeddings;
        else delete intermediates.pooled;
        out.intermediates = { tokens, sequence_id: ids, ...intermediates };
      }
      return out;
    } finally {
      await this.end();
    }
  }
  async summarize(logp, { signal } = {}) {
    signal?.throwIfAborted();
    this.begin();
    try {
      const b = this.buffer(logp, true),
        out = await this.dispatch(summarySource(logp.length), [b], 2, [1]);
      const { scores } = await this.readOutputs(
        { scores: { buffer: out, n: 2 } },
        signal,
      );
      return { naturalness: scores[0], pseudoPerplexity: scores[1] };
    } finally {
      await this.end();
    }
  }
  async run(sequence, options) {
    if (this.taskBusy) throw new Error("An inference task is already running");
    this.taskBusy = true;
    try {
      return await runTask(this, sequence, options);
    } finally {
      this.taskBusy = false;
    }
  }
  async batch(sequences, options) {
    const out = [];
    for (const s of sequences) out.push(await this.run(s, options));
    return out;
  }
  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const b of this.buffers) b.destroy();
    this.buffers.clear();
    this.weights = {};
    this.pipelines.clear();
    if (this.ownsDevice) this.device.destroy();
  }
}
export async function createEsmcEngine({
  baseUrl,
  precision = "int8",
  device,
  onProgress,
  signal,
  useF16 = true,
  cache = "cache",
  read,
} = {}) {
  if (!baseUrl)
    throw new Error("baseUrl must contain fp32/, fp16/, int8/ bundles");
  if (!["fp32", "fp16", "int8"].includes(precision))
    throw new Error("Unknown precision");
  signal?.throwIfAborted();
  const ownsDevice = !device;
  let info;
  if (!device) ({ device, info } = await requestWebGpuDevice({ useF16 }));
  const engine = new Engine(device, { ownsDevice, useF16 });
  engine.adapterInfo = info;
  try {
    const bundle = await loadWeights(
      `${baseUrl.replace(/\/$/, "")}/${precision}`,
      {
        onProgress,
        signal,
        cache,
        read,
        onTensor: (name, t) => engine.upload(name, t),
      },
    );
    engine.config = bundle.config;
    engine.manifest = bundle.manifest;
    return engine;
  } catch (e) {
    engine.destroy();
    throw e;
  }
}
