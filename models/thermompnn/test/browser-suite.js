import { topkShader } from "../webgpu/shaders.js";
import { createThermoMPNNEngine, assertHardware } from "../webgpu/index.js";
const assert = (v, m) => {
  if (!v) throw new Error(m);
};
export function relRMS(a, b) {
  assert(a.length === b.length, "Length mismatch");
  let e = 0,
    r = 0,
    maxAbs = 0;
  for (let i = 0; i < a.length; i++) {
    assert(Number.isFinite(a[i]), "Nonfinite result");
    e += (a[i] - b[i]) ** 2;
    r += b[i] ** 2;
    maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
  }
  return { relRMS: Math.sqrt(e / Math.max(r, 1e-30)), maxAbs };
}
export async function run() {
  const adapter = await navigator.gpu?.requestAdapter();
  const info = assertHardware(adapter);
  const metrics = {
    adapter: {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description,
    },
    precisions: {},
  };
  console.log("Hardware:", JSON.stringify(metrics.adapter));
  const cases = await (await fetch("../fixtures/cases.json")).json();
  for (const [precision, useF16] of [
    ["fp32", false],
    ["fp16", false],
    ["fp16", true],
    ["int8", false],
  ]) {
    const label = precision + (useF16 ? "-native" : "");
    const start = performance.now(),
      engine = await createThermoMPNNEngine({
        baseUrl: location.origin + "/weights/default",
        precision,
        useF16,
      });
    if (useF16) assert(engine.useF16, "Native shader-f16 was not enabled");
    const result = { coldLoadMs: performance.now() - start, cases: {} };
    metrics.precisions[label] = result;
    try {
      for (const c of cases) {
        const text = await (await fetch("../fixtures/" + c.input)).text(),
          options = {
            chains: c.inputChains ?? c.chains,
            returnIntermediates: true,
          };
        const actual = await engine.predict(text, options);
        const stem = `../fixtures/oracle_${c.same_as ?? c.id}_${precision}`;
        const meta = await (await fetch(stem + ".json")).json(),
          binary = await (await fetch(stem + ".bin")).arrayBuffer();
        const traces = {};
        for (const [name, t] of Object.entries(meta.tensors)) {
          const expected = new (t.dtype === "I32" ? Int32Array : Float32Array)(
              binary,
              t.offset,
              t.byteLength / 4,
            ),
            got = actual.intermediates[name];
          assert(got, `Missing ${name}`);
          const m = relRMS(got, expected);
          traces[name] = m;
          assert(
            m.relRMS < 1e-4,
            `${label}/${c.id}/${name}: ${JSON.stringify(m)}`,
          );
          if (t.dtype === "I32")
            assert(
              got.every((v, i) => v === expected[i]),
              `Integer trace ${name} mismatch`,
            );
        }
        assert(
          engine.memoryUsage().transientBuffers === 0,
          "Transient buffer leak",
        );
        const r = actual.residues.find((r) => r.aa !== "X");
        const targeted = await engine.predict(text, {
          chains: options.chains,
          mode: "targeted",
          mutations: [
            {
              chain: r.chain,
              number: r.number,
              ins: r.ins,
              wildtype: r.aa,
              mutation: "A",
            },
            { index: r.index, mutation: r.aa },
          ],
        });
        assert(
          targeted.scores[0] === actual.ddg[r.index * 20] &&
            targeted.scores[1] === 0,
          "Targeted mismatch",
        );
        result.cases[c.id] = { timingMs: actual.timingMs, traces };
        console.log(
          `${label}/${c.id}: ${traces.ddg.relRMS.toExponential(3)} (${actual.timingMs.toFixed(1)} ms)`,
        );
      }
      result.benchmarks = {};
      for (const name of ["small", "medium", "full"]) {
        const text = await (await fetch("../fixtures/" + name + ".pdb")).text();
        const warmup = await engine.predict(text);
        if (name === "full") {
          const stem = "../fixtures/oracle_full_length_" + precision,
            meta = await (await fetch(stem + ".json")).json(),
            binary = await (await fetch(stem + ".bin")).arrayBuffer(),
            t = meta.tensors.ddg;
          const m = relRMS(
            warmup.ddg,
            new Float32Array(binary, t.offset, t.byteLength / 4),
          );
          assert(m.relRMS < 1e-4, "Full chain GPU parity");
          result.fullLengthParity = m;
        }
        const samples = [];
        for (let i = 0; i < 5; i++)
          samples.push((await engine.predict(text)).timingMs);
        result.benchmarks[name] = {
          residues: name === "small" ? 12 : name === "medium" ? 76 : 194,
          samplesMs: samples,
          medianMs: [...samples].sort((a, b) => a - b)[2],
        };
      }
    } finally {
      engine.destroy();
      engine.destroy();
      assert(engine.memoryUsage().totalBytes === 0, "Resident buffer leak");
    }
  }
  metrics.topk = await testTopk();
  metrics.lifecycle = await lifecycle(cases);
  metrics.storageFallback = await storageFallback();
  return metrics;
}

async function rejects(fn, pattern) {
  let error;
  try {
    await fn();
  } catch (e) {
    error = e;
  }
  assert(error, "Expected rejection");
  if (pattern) assert(pattern.test(error.message), error.message);
  return error;
}
async function lifecycle() {
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  assertHardware(adapter);
  const device = await adapter.requestDevice();
  const uncaptured = [];
  device.addEventListener("uncapturederror", (e) =>
    uncaptured.push(e.error.message),
  );
  const engine = await createThermoMPNNEngine({
    baseUrl: location.origin + "/weights/default",
    device,
  });
  const text = await (await fetch("../fixtures/small.pdb")).text(),
    clean = () =>
      assert(
        engine.memoryUsage().transientBuffers === 0 &&
          engine.memoryUsage().transientBytes === 0,
        "Transient leak after failure",
      );
  try {
    await rejects(() => engine.predict(""), /empty/);
    clean();
    await rejects(() => engine.predict(text, { mode: "double" }), /mode/);
    clean();
    await rejects(
      () =>
        engine.predict(text, {
          mode: "targeted",
          mutations: [{ index: 0, mutation: "X" }],
        }),
      /canonical/,
    );
    clean();
    const before = new AbortController();
    before.abort();
    await rejects(() => engine.predict(text, { signal: before.signal }));
    clean();
    const during = new AbortController();
    await rejects(() =>
      engine.predict(text, {
        signal: during.signal,
        onProgress: () => during.abort(),
      }),
    );
    clean();
    await rejects(
      () =>
        engine.predict(text, {
          onProgress: () => {
            throw new Error("Injected callback failure");
          },
        }),
      /Injected/,
    );
    clean();
    const running = engine.predict(text);
    await rejects(() => engine.predict(text), /progress/);
    await running;
    clean();
    const timed = new AbortController();
    const promise = engine.predict(text, { signal: timed.signal });
    setTimeout(() => timed.abort(), 0);
    await rejects(() => promise);
    clean();
    // Buffer limit admission with a genuine WebGPU device, and a wrapper exposing stricter limits.
    const limited = new Proxy(device, {
      get(target, key) {
        if (key === "limits")
          return {
            maxBufferSize: 4_000_000,
            maxStorageBufferBindingSize: 4_000_000,
          };
        const value = Reflect.get(target, key, target);
        return typeof value === "function" ? value.bind(target) : value;
      },
    });
    const e = await createThermoMPNNEngine({
      baseUrl: location.origin + "/weights/default",
      device: limited,
    });
    try {
      const medium = await (await fetch("../fixtures/medium.pdb")).text();
      await rejects(() => e.predict(medium), /admission/);
      assert(
        e.memoryUsage().transientBuffers === 0,
        "Admission allocated temporary buffers",
      );
    } finally {
      e.destroy();
    }
    // Throw during the input-buffer constructor to verify partial-allocation cleanup.
    let allocations = 0,
      failAt = Infinity;
    const faultDevice = new Proxy(device, {
      get(target, key) {
        if (key === "createBuffer")
          return (descriptor) => {
            if (++allocations === failAt)
              throw new Error("Injected allocation failure");
            return target.createBuffer(descriptor);
          };
        const value = Reflect.get(target, key, target);
        return typeof value === "function" ? value.bind(target) : value;
      },
    });
    const faultEngine = await createThermoMPNNEngine({
      baseUrl: location.origin + "/weights/default",
      device: faultDevice,
    });
    try {
      failAt = allocations + 3;
      await rejects(() => faultEngine.predict(text), /allocation failure/);
      assert(
        faultEngine.memoryUsage().transientBuffers === 0,
        "Partially constructed operation leaked",
      );
      failAt = Infinity;
      await faultEngine.predict(text);
    } finally {
      faultEngine.destroy();
    }
    await engine.predict(text);
    clean();
  } finally {
    engine.destroy();
    engine.destroy();
  }
  assert(engine.memoryUsage().totalBytes === 0, "Destroy leaked buffers");
  await rejects(() => engine.predict(text), /destroyed/);
  const b = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(b, 0, new Uint32Array([1]));
  await device.queue.onSubmittedWorkDone();
  b.destroy();
  assert(!uncaptured.length, "Uncaptured GPU errors: " + uncaptured.join(";"));
  device.destroy();
  // Loading cancellation and inference teardown on self-managed devices.
  const cancel = new AbortController();
  await rejects(() =>
    createThermoMPNNEngine({
      baseUrl: location.origin + "/weights/default",
      signal: cancel.signal,
      onProgress: () => cancel.abort(),
    }),
  );
  const doomed = await createThermoMPNNEngine({
    baseUrl: location.origin + "/weights/default",
  });
  const inFlight = doomed.predict(text, { onProgress: () => doomed.destroy() });
  await rejects(() => inFlight);
  assert(
    doomed.memoryUsage().totalBytes === 0,
    "Destroy during inference leaked",
  );
  return {
    success: true,
    preAbort: true,
    midEncodingAbort: true,
    eventLoopAbort: true,
    callbackError: true,
    partialAllocationFailure: true,
    overlapRejected: true,
    admissionRejected: true,
    externalDeviceSurvives: true,
    destroyDuringInference: true,
    loadAbort: true,
    uncapturedErrors: uncaptured,
  };
}
async function storageFallback() {
  const text = await (await fetch("../fixtures/small.pdb")).text(),
    results = {};
  await caches.delete("thermompnn-v1");
  const root = await navigator.storage.getDirectory();
  try {
    await root.removeEntry("thermompnn-v1", { recursive: true });
  } catch (e) {
    if (e.name !== "NotFoundError") throw e;
  }
  for (const [kind, method, errorName] of [
    ["cache", "open", "SecurityError"],
    ["cache", "put", "QuotaExceededError"],
    ["opfs", "getDirectory", "SecurityError"],
    ["opfs", "createWritable", "QuotaExceededError"],
  ]) {
    const target =
      method === "open"
        ? CacheStorage.prototype
        : method === "put"
          ? Cache.prototype
          : method === "getDirectory"
            ? StorageManager.prototype
            : FileSystemFileHandle.prototype;
    const descriptor = Object.getOwnPropertyDescriptor(target, method),
      original = target[method];
    let calls = 0;
    Object.defineProperty(target, method, {
      configurable: true,
      writable: true,
      value: async function () {
        calls++;
        throw new DOMException("Private storage unavailable", errorName);
      },
    });
    let engine;
    try {
      engine = await createThermoMPNNEngine({
        baseUrl: location.origin + "/weights/default",
        precision: "fp32",
        cache: kind,
      });
      const out = await engine.predict(text);
      assert(out.ddg.every(Number.isFinite), "Storage fallback result");
      assert(calls > 0, `Storage fault not exercised: ${method}`);
      results[kind + "-" + method] = { passed: true, calls };
    } finally {
      engine?.destroy();
      if (descriptor) Object.defineProperty(target, method, descriptor);
      else target[method] = original;
    }
  }
  return results;
}

async function testTopk() {
  const cases = await (await fetch("../fixtures/topk_cases.json")).json(),
    adapter = await navigator.gpu.requestAdapter();
  assertHardware(adapter);
  const device = await adapter.requestDevice();
  try {
    for (const c of cases) {
      const L = c.values.length,
        K = Math.min(L, 48),
        d = device.createBuffer({
          size: L * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        }),
        out = device.createBuffer({
          size: K * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        }),
        staging = device.createBuffer({
          size: K * 4,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
      device.pushErrorScope("validation");
      try {
        device.queue.writeBuffer(d, 0, new Float32Array(c.values));
        const module = device.createShaderModule({ code: topkShader(L, K) }),
          p = device.createComputePipeline({
            layout: "auto",
            compute: { module, entryPoint: "main" },
          }),
          group = device.createBindGroup({
            layout: p.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: d } },
              { binding: 1, resource: { buffer: out } },
            ],
          });
        const encoder = device.createCommandEncoder(),
          pass = encoder.beginComputePass();
        pass.setPipeline(p);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(1);
        pass.end();
        encoder.copyBufferToBuffer(out, 0, staging, 0, K * 4);
        device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const got = new Uint32Array(staging.getMappedRange()).slice();
        staging.unmap();
        assert(
          got.every((v, i) => v === c.indices[i]),
          `GPU topk tie order: ${c.name}`,
        );
      } finally {
        d.destroy();
        out.destroy();
        staging.destroy();
        const error = await device.popErrorScope();
        assert(!error, error?.message);
      }
    }
  } finally {
    device.destroy();
  }
  return { passed: cases.length };
}
