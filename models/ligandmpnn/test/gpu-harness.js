import {
  requestWebGpuDevice,
  GpuOps,
  LigandModel,
  linearShader,
  normShader,
  elementShader,
  GELU_BODY,
  createLigandMpnnEngine,
} from "/webgpu/index.js";
import {
  fetchWeights,
  parseStructure,
  featurize,
} from "/shared/ligandmpnn/index.js";
const log = (msg) => fetch("/log", { method: "POST", body: String(msg) });
const assert = (x, msg) => {
  if (!x) throw Error(msg);
};
function stats(got, ref) {
  assert(got.length === ref.length, "Length mismatch");
  let e = 0,
    r = 0,
    max = 0;
  for (let i = 0; i < got.length; i++) {
    const d = got[i] - ref[i];
    e += d * d;
    r += ref[i] ** 2;
    max = Math.max(max, Math.abs(d));
  }
  return { relRMS: Math.sqrt(e / Math.max(r, 1e-30)), maxAbs: max };
}
async function run() {
  const { adapter, device, hasF16 } = await requestWebGpuDevice(),
    ai = adapter.info,
    adapterInfo = {
      vendor: ai.vendor,
      architecture: ai.architecture,
      device: ai.device,
      description: ai.description,
      isFallbackAdapter: ai.isFallbackAdapter,
    };
  await log(JSON.stringify({ adapterInfo, hasF16 }));
  assert(!ai.isFallbackAdapter, "Hardware adapter required");
  const errors = [];
  device.addEventListener("uncapturederror", (e) =>
    errors.push(e.error.message),
  );
  device.pushErrorScope("validation");
  if (new URL(import.meta.url).searchParams.has("shaders")) {
    const shaders = [
      linearShader(3, 33, 21),
      linearShader(3, 33, 21, { bias: false }),
      linearShader(3, 33, 21, { packed: true }),
      normShader(3, 128),
      elementShader(["x", "out"], GELU_BODY, 128),
    ];
    if (hasF16)
      shaders.push(linearShader(3, 33, 21, { packed: true, f16: true }));
    for (const code of shaders) {
      const module = device.createShaderModule({ code });
      const info = await module.getCompilationInfo();
      assert(
        !info.messages.some((m) => m.type === "error"),
        JSON.stringify(info.messages),
      );
      await device.createComputePipelineAsync({
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
    }
    const ops = new GpuOps(device);
    for (const width of [1, 7, 128, 259]) {
      const rows = 3;
      const values = Float32Array.from({ length: rows * width }, (_, i) =>
        i < width ? 4 : Math.sin(i * 0.3) + (i % 9));
      const scale = Float32Array.from({ length: width }, (_, i) => 0.5 + i % 3);
      const bias = Float32Array.from({ length: width }, (_, i) => i % 5 - 2);
      const x = ops.tensor(values, [rows, width]);
      const w = ops.tensor(scale, [width]);
      const b = ops.tensor(bias, [width]);
      device.pushErrorScope("validation");
      const result = await ops.read(ops.norm(x, w, b));
      const error = await device.popErrorScope();
      assert(!error, error?.message);
      for (let row = 0; row < rows; row++) {
        const src = values.subarray(row * width, (row + 1) * width);
        const mean = src.reduce((s, v) => s + v, 0) / width;
        const variance = src.reduce((s, v) => s + (v - mean) ** 2, 0) / width;
        for (let col = 0; col < width; col++) {
          const expected = (src[col] - mean) / Math.sqrt(variance + 1e-5) * scale[col] + bias[col];
          if (!(Math.abs(result[row * width + col] - expected) < 2e-5)) {
            throw Error("LayerNorm mismatch at width " + width + ", row " + row + ", column " + col + ": got " + result[row * width + col] + ", expected " + expected);
          }
        }
      }
    }
    ops.destroy();
    const scope = await device.popErrorScope();
    assert(!scope, scope?.message);
    device.destroy();
    return { ok: true, adapterInfo, hasF16, shaders: shaders.length };
  }
  const meta = await (await fetch("/fixtures/oracle_ligandmpnn.json")).json(),
    bin = await (await fetch("/fixtures/oracle_ligandmpnn.bin")).arrayBuffer(),
    read = (name) => {
      const r = meta.tensors[name];
      assert(r, "Missing oracle " + name);
      const b = bin.slice(r.byteOffset, r.byteOffset + r.byteLength);
      return r.dtype === "I32" ? new Int32Array(b) : new Float32Array(b);
    };
  const text = await (await fetch("/fixtures/holo_1BC8.pdb")).text(),
    parsed = parseStructure(text),
    input = featurize(parsed, meta.options),
    randn = read("input.randn"),
    result = { ok: true, adapterInfo, hasF16, cases: {} };
  const bundle = await fetchWeights("/weights/ligandmpnn_v_32_010_25/fp32", {
      cache: false,
    }),
    ops = new GpuOps(device),
    model = new LigandModel(ops, bundle);
  await log("FP32 score starting");
  const scored = await model.score(input, { randn, trace: true });
  const initialScope = await device.popErrorScope();
  assert(!initialScope, initialScope?.message);
  device.pushErrorScope("validation");
  let worst = { name: "", relRMS: 0 };
  for (const [key, value] of Object.entries(scored.trace)) {
    const s = stats(value, read("score." + key));
    assert(s.relRMS < 1e-4, `FP32 ${key}: ${JSON.stringify(s)}`);
    if (s.relRMS > worst.relRMS) worst = { name: key, ...s };
  }
  result.score = {
    worst,
    logits: stats(scored.logits, read("score.output.logits")),
  };
  ops.clear();
  await log("FP32 score " + JSON.stringify(result.score));
  for (const which of ["sample", "tied", "sidechain", "no_context"]) {
    const opts = { ...meta.options };
    if (which === "tied")
      opts.tiedPositions = [
        [2, 3],
        [4, 5],
      ];
    if (which === "sidechain") opts.useSideChainContext = true;
    if (which === "no_context") opts.useAtomContext = false;
    const start = performance.now(),
      out = await model.sample(featurize(parsed, opts), {
        randn,
        uniforms: read(which + ".uniforms"),
        temperature: 0.3,
        trace: true,
      });
    let worst = { name: "", relRMS: 0 };
    for (const [key, value] of Object.entries(out.trace)) {
      if (key.startsWith("encoded.")) continue;
      const s = stats(value, read(which + "." + key));
      assert(s.relRMS < 1e-4, `${which} ${key}: ${JSON.stringify(s)}`);
      if (s.relRMS > worst.relRMS) worst = { name: key, ...s };
    }
    assert(
      out.S.every((a, i) => a === read(which + ".output.S")[i]),
      which + " sequence mismatch",
    );
    for (const key of ["sampling_probs", "log_probs"])
      assert(
        stats(out[key], read(which + ".output." + key)).relRMS < 1e-4,
        which + " " + key,
      );
    for (const key of ["global_score", "ligand_score"])
      assert(
        stats([out[key]], read(which + "." + key)).relRMS < 1e-4,
        which + " " + key,
      );
    result.cases[which] = {
      worst,
      sequence: out.sequence,
      ms: performance.now() - start,
    };
    ops.clear();
    await log(which + " " + JSON.stringify(result.cases[which]));
  }
  const timings = [];
  for (let i = 0; i < 4; i++) {
    const t = performance.now();
    await model.sample(input, {
      randn,
      uniforms: read("sample.uniforms"),
      temperature: 0.3,
    });
    timings.push(performance.now() - t);
    ops.clear();
  }
  result.benchmark = {
    coldOrWarmupMs: timings[0],
    warmMs: timings.slice(1),
    meanWarmMs: timings.slice(1).reduce((a, b) => a + b, 0) / 3,
    peakBufferBytes: ops.peakBytes,
  };
  result.shaders = await ops.checkShaders();
  ops.destroy();
  assert(ops.buffers.size === 0 && ops.bytes === 0, "Leaked buffers");
  const qm = await (await fetch("/fixtures/oracle_precisions.json")).json(),
    qb = await (await fetch("/fixtures/oracle_precisions.bin")).arrayBuffer(),
    qread = (name) => {
      const r = qm.tensors[name];
      return new Float32Array(
        qb.slice(r.byteOffset, r.byteOffset + r.byteLength),
      );
    };
  const halfBundle = await fetchWeights(
    "/weights/ligandmpnn_v_32_010_25/fp16",
    { cache: false },
  );
  const halfResults = {};
  for (const useF16 of [false, ...(hasF16 ? [true] : [])]) {
    const halfOps = new GpuOps(device, { half: true, useF16 }),
      halfModel = new LigandModel(halfOps, halfBundle);
    const out = await halfModel.score(input, { randn, trace: true });
    // Explicitly report the measured weight rounding floor; never relabel it as <5e-4.
    const rounded = stats(out.logits, qread("010.fp16.logits"));
    assert(
      rounded.relRMS < 1e-4,
      "FP16 runtime parity " + JSON.stringify(rounded),
    );
    let roundedWorst = { name: "", relRMS: 0 };
    for (const [k, v] of Object.entries(out.trace)) {
      if (!qm.tensors["010.fp16." + k]) continue;
      const d = stats(v, qread("010.fp16." + k));
      assert(d.relRMS < 1e-4, "FP16 " + k + " " + JSON.stringify(d));
      if (d.relRMS > roundedWorst.relRMS) roundedWorst = { name: k, ...d };
    }
    const s = stats(out.logits, read("score.output.logits"));
    halfResults[useF16 ? "shaderF16" : "core"] = {
      runtimeLogits: rounded,
      runtimeWorst: roundedWorst,
      logitsVsFp32: s,
      logProbsVsFp32: stats(out.log_probs, read("score.output.log_probs")),
      requestedLogitsToleranceMet: s.relRMS < 5e-4,
    };
    const replay = await halfModel.sample(input, {
      randn,
      uniforms: read("sample.uniforms"),
      temperature: 0.3,
    });
    assert(
      replay.S.every((a, i) => a === read("sample.output.S")[i]),
      "FP16 sequence replay mismatch",
    );
    result.shaders += await halfOps.checkShaders();
    halfOps.destroy();
    await log(
      "FP16 " + JSON.stringify(halfResults[useF16 ? "shaderF16" : "core"]),
    );
  }
  result.fp16 = halfResults;
  result.otherBundles = {};
  for (const [variant, precision] of [
    ["010", "int8"],
    ["020", "fp16"],
    ["020", "int8"],
  ]) {
    const b = await fetchWeights(
        "/weights/ligandmpnn_v_32_" + variant + "_25/" + precision,
        { cache: false },
      ),
      o = new GpuOps(device, { half: precision === "fp16", useF16: hasF16 }),
      m = new LigandModel(o, b),
      out = await m.score(input, { randn, trace: true });
    let worst = { name: "", relRMS: 0 };
    for (const [k, v] of Object.entries(out.trace)) {
      if (!qm.tensors[variant + "." + precision + "." + k]) continue;
      const d = stats(v, qread(variant + "." + precision + "." + k));
      assert(
        d.relRMS < 1e-4,
        variant + " " + precision + " " + k + " " + JSON.stringify(d),
      );
      if (d.relRMS > worst.relRMS) worst = { name: k, ...d };
    }
    const logits = stats(
      out.logits,
      qread(variant + "." + precision + ".logits"),
    );
    assert(logits.relRMS < 1e-4, "Rounded bundle logits");
    result.otherBundles[variant + "." + precision] = { worst, logits };
    result.shaders += await o.checkShaders();
    o.destroy();
  }
  const sm = await (await fetch("/fixtures/oracle_small_molecule.json")).json(),
    sb = await (
      await fetch("/fixtures/oracle_small_molecule.bin")
    ).arrayBuffer(),
    sr = (name) => {
      const r = sm.tensors[name],
        b = sb.slice(r.byteOffset, r.byteOffset + r.byteLength);
      return r.dtype === "F32" ? new Float32Array(b) : new Int32Array(b);
    },
    st = await (await fetch("/fixtures/" + sm.structure)).text();
  const so = new GpuOps(device),
    mod = new LigandModel(so, bundle),
    si = featurize(parseStructure(st), sm.options),
    sc = await mod.score(si, { randn: sr("input.randn") }),
    ss = stats(sc.logits, sr("score.logits"));
  assert(ss.relRMS < 1e-4, "Small molecule logits " + JSON.stringify(ss));
  const sampled = await mod.sample(si, {
    randn: sr("input.randn"),
    uniforms: sr("uniforms"),
    temperature: 0.3,
  });
  assert(
    sampled.S.every((a, i) => a === sr("sample.S")[i]),
    "Small molecule sequence",
  );
  assert(
    stats(sampled.log_probs, sr("sample.log_probs")).relRMS < 1e-4,
    "Small molecule log probs",
  );
  result.smallMolecule = { logits: ss, sequence: sampled.sequence };
  result.shaders += await so.checkShaders();
  so.destroy();

  const engine = await createLigandMpnnEngine({ baseUrl: "/weights", device }),
    designed = await engine.design(text, {
      ...meta.options,
      seed: 42,
      numSequences: 2,
    });
  assert(designed.sequences.length === 2, "Public API numSequences");
  assert(
    designed.sequences.every((s) => s.startsWith("WQ") && !s.includes("C")),
    "Public constraints",
  );
  const fullText = await (await fetch("/inputs/1BC8.pdb")).text();
  const fullCold = await engine.design(fullText, {
    temperature: 0.3,
    seed: 42,
  });
  const fullWarm = await engine.design(fullText, {
    temperature: 0.3,
    seed: 42,
  });
  assert(fullWarm.sequences[0].length === 93, "Full 1BC8 length");
  assert(
    fullWarm.sequences[0] === fullCold.sequences[0],
    "Seeded full-structure replay",
  );
  result.fullStructure = {
    length: 93,
    coldMs: fullCold.timingMs,
    warmMs: fullWarm.timingMs,
    sequence: fullWarm.sequences[0],
  };
  result.shaders += await engine.model.o.checkShaders();
  engine.destroy();
  result.api = { sequences: designed.sequences, timingMs: designed.timingMs };
  const worker = new Worker("/demo/worker.js", { type: "module" });
  try {
    const wr = await new Promise((resolve, reject) => {
      worker.onerror = (e) => reject(Error(e.message));
      worker.onmessage = ({ data }) => {
        if (data.type === "result") resolve(data.result);
        if (data.type === "error") reject(Error(data.message));
      };
      worker.postMessage({
        type: "design",
        config: { baseUrl: new URL("/weights", location.href).href },
        structure: text,
        options: { ...meta.options, numSequences: 1, seed: 42 },
      });
    });
    assert(
      wr.sequences[0] === designed.sequences[0],
      "Worker API deterministic parity",
    );
    result.worker = { sequence: wr.sequences[0], timingMs: wr.timingMs };
  } finally {
    worker.postMessage({ type: "destroy" });
    worker.terminate();
  }

  const scope = await device.popErrorScope();
  assert(!scope, scope?.message);
  assert(!errors.length, errors.join("\n"));
  result.validationErrors = errors;
  device.destroy();
  return result;
}
run()
  .then((r) => fetch("/done", { method: "POST", body: JSON.stringify(r) }))
  .catch((e) =>
    fetch("/done", {
      method: "POST",
      body: JSON.stringify({ ok: false, error: e.stack ?? String(e) }),
    }),
  );
