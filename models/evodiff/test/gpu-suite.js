import { createEvoDiffEngine, rejectSoftware } from "../webgpu/index.js";
import { oracle, checkEngine, assert, parity } from "./parity.js";
const read = async (url) => {
  const r = await fetch("/" + url);
  if (!r.ok) throw new Error(`Missing fixture ${url}`);
  return r.arrayBuffer();
};
const log = (message) =>
  fetch("/log", { method: "POST", body: String(message) });
export async function run() {
  assert(navigator.gpu, "Hardware WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  assert(adapter, "Adapter missing");
  rejectSoftware(adapter.info);
  const info = {
    vendor: adapter.info.vendor,
    architecture: adapter.info.architecture,
    device: adapter.info.device,
    description: adapter.info.description,
    isFallbackAdapter: adapter.info.isFallbackAdapter,
  };
  assert(
    /nvidia/i.test(info.vendor + " " + info.description),
    "Expected real NVIDIA hardware",
  );
  const device = await adapter.requestDevice({
    requiredFeatures: adapter.features.has("shader-f16") ? ["shader-f16"] : [],
  });
  const results = {
    adapter: info,
    userAgent: navigator.userAgent,
    skipped: 0,
    precisions: {},
  };
  try {
    results.initializationCleanup = await initializationCleanup(device);
    for (const precision of ["fp32", "fp16", "int8"]) {
      const ref = await oracle(precision, read),
        start = performance.now();
      const engine = await createEvoDiffEngine({
        baseUrl: location.origin + "/weights/oa_dm_38m",
        precision,
        device,
        cache: "none",
      });
      const coldLoadMs = performance.now() - start;
      try {
        results.precisions[precision] = {
          coldLoadMs,
          ...(await checkEngine(engine, ref, log)),
        };
        results.precisions[precision].memory = engine.memoryUsage();
        assert(
          engine.memoryUsage().transientBuffers === 0,
          "Normal transient leak",
        );
        await hygiene(engine, device);
        const benchmarks = {};
        for (const L of [32, 128, 256]) {
          await engine.forward("#".repeat(L));
          const runs = [];
          for (let i = 0; i < 3; i++)
            runs.push((await engine.forward("#".repeat(L))).timingMs);
          benchmarks[L] = {
            runsMs: runs,
            medianMs: runs.slice().sort((a, b) => a - b)[1],
          };
        }
        results.precisions[precision].benchmarks = benchmarks;
      } finally {
        engine.destroy();
        engine.destroy();
        assert(engine.memoryUsage().totalBytes === 0, "Destroy leaked");
      }
    }
    assert(
      device.features.has("shader-f16"),
      "Native F16 test requires shader-f16",
    );
    const engine = await createEvoDiffEngine({
      baseUrl: location.origin + "/weights/oa_dm_38m",
      precision: "fp16",
      device,
      useF16: true,
      cache: "none",
    });
    try {
      const ref = await oracle("fp16", read),
        c = ref.meta.A[0];
      results.nativeF16 = parity(
        (await engine.forward(c.sequence)).logits,
        ref.tensor(c.trace.logits),
        "native f16",
      );
    } finally {
      engine.destroy();
    }
  } finally {
    device.destroy();
  }
  results.workerStorage = await workerStorage();
  results.demo = await demo();
  return results;
}
async function rejects(fn) {
  let failed = false;
  try {
    await fn();
  } catch {
    failed = true;
  }
  assert(failed, "Expected an error");
}
async function hygiene(engine, device) {
  await rejects(() => engine.forward("INVALID?"));
  assert(engine.memoryUsage().transientBuffers === 0, "Error leak");
  const submit = device.queue.submit;
  try {
    device.queue.submit = () => {
      throw new Error("Injected submit failure with live readback buffer");
    };
    await rejects(() => engine.forward("MSKGEELF"));
  } finally {
    device.queue.submit = submit;
  }
  assert(
    engine.memoryUsage().transientBuffers === 0,
    "Thrown error leaked readback buffer",
  );
  await rejects(() =>
    engine.forward(
      new Int32Array(
        Math.floor(device.limits.maxStorageBufferBindingSize / (1024 * 4)) + 1,
      ),
    ),
  );
  const controller = new AbortController();
  controller.abort();
  await rejects(() =>
    engine.forward("MSKGEELF", { signal: controller.signal }),
  );
  const live = new AbortController();
  const promise = engine.forward("#".repeat(128), { signal: live.signal });
  setTimeout(() => live.abort(), 0);
  await rejects(() => promise);
  assert(engine.memoryUsage().transientBuffers === 0, "Abort leaked");
  await rejects(() => engine.generate({ length: 4, order: [0, 0, 1, 2] }));
  const during = new AbortController();
  await rejects(() =>
    engine.generate({
      length: 4,
      signal: during.signal,
      onProgress() {
        during.abort();
      },
    }),
  );
  assert(engine.memoryUsage().transientBuffers === 0, "Sampling abort leaked");
  await engine.forward("MSKGEELF");
  const a = await engine.generate({ length: 4, seed: 991 }),
    b = await engine.generate({ length: 4, seed: 991 });
  assert(a.sequence === b.sequence, "Seed replay mismatch");
}
async function initializationCleanup(device) {
  const create = device.createBuffer,
    allocated = new Set();
  device.createBuffer = function (...args) {
    const b = create.apply(this, args),
      destroy = b.destroy;
    allocated.add(b);
    b.destroy = function () {
      allocated.delete(b);
      return destroy.call(this);
    };
    return b;
  };
  try {
    const abort = new AbortController();
    await rejects(() =>
      createEvoDiffEngine({
        baseUrl: location.origin + "/weights/oa_dm_38m",
        precision: "int8",
        device,
        cache: "none",
        signal: abort.signal,
        onProgress: (p) => {
          if (p.phase === "weights") abort.abort();
        },
      }),
    );
    assert(allocated.size === 0, "Initialization abort leaked weights");
    await rejects(() =>
      createEvoDiffEngine({
        baseUrl: location.origin + "/weights/oa_dm_38m",
        precision: "int8",
        device,
        cache: "none",
        onProgress() {
          throw new Error("Injected load callback failure");
        },
      }),
    );
    assert(allocated.size === 0, "Initialization error leaked weights");
    return { abortedBuffers: 0, errorBuffers: 0 };
  } finally {
    device.createBuffer = create;
    for (const b of allocated) b.destroy();
  }
}
async function workerStorage() {
  const worker = new Worker(new URL("./worker.js", import.meta.url), {
    type: "module",
  });
  let id = 0;
  const waiting = new Map();
  worker.onmessage = ({ data }) => {
    const item = waiting.get(data.id);
    waiting.delete(data.id);
    if (data.error) item.reject(new Error(data.error));
    else item.resolve(data.result);
  };
  worker.onerror = (e) => {
    for (const p of waiting.values()) p.reject(new Error(e.message));
    waiting.clear();
  };
  const call = (cmd, args) =>
    new Promise((resolve, reject) => {
      const n = ++id;
      waiting.set(n, { resolve, reject });
      worker.postMessage({ id: n, cmd, args });
    });
  const result = {};
  try {
    const ref = await oracle("int8", read),
      c = ref.meta.A[0];
    for (const cache of ["cache", "opfs"]) {
      await call("init", { cache });
      await call("forward", { sequence: c.sequence });
      await call("destroy");
      await call("offline", { value: true });
      await call("init", { cache });
      const out = await call("forward", { sequence: c.sequence });
      result[cache] = {
        offline: true,
        ...parity(out.logits, ref.tensor(c.trace.logits), "worker offline"),
        memory: out.memory,
      };
      assert(out.memory.transientBuffers === 0, "Worker leak");
      assert((await call("destroy")).totalBytes === 0, "Worker destroy leak");
      await call("offline", { value: false });
    }
    for (const cache of ["cache", "opfs"]) {
      await call("privateMode", { cache });
      const out = await call("forward", { sequence: c.sequence });
      result[cache + "PrivateFallback"] = parity(
        out.logits,
        ref.tensor(c.trace.logits),
        "Private fallback",
      );
      await call("destroy");
    }
  } finally {
    worker.terminate();
  }
  return result;
}
async function demo() {
  const frame = document.createElement("iframe");
  frame.src = "/demo/index.html";
  document.body.appendChild(frame);
  await new Promise((resolve, reject) => {
    frame.onload = resolve;
    frame.onerror = reject;
  });
  const doc = frame.contentDocument,
    win = frame.contentWindow,
    $ = (id) => doc.getElementById(id);
  async function until(fn) {
    const start = performance.now();
    while (!fn()) {
      if (performance.now() - start > 120000)
        throw new Error("Demo timeout: " + $("status").textContent);
      if ($("status").textContent.startsWith("Error:"))
        throw new Error($("status").textContent);
      await new Promise((r) => setTimeout(r, 50));
    }
  }
  try {
    $("length").value = "8";
    $("run").click();
    await until(() => $("status").textContent.startsWith("Done."));
    assert($("sequence").textContent.length === 8, "Demo unconditional length");
    let blob;
    const create = win.URL.createObjectURL,
      click = win.HTMLAnchorElement.prototype.click;
    win.URL.createObjectURL = (b) => {
      blob = b;
      return create.call(win.URL, b);
    };
    win.HTMLAnchorElement.prototype.click = () => {};
    try {
      $("download").click();
      assert(
        (await blob.text()).startsWith(">EvoDiff_OA_DM_38M"),
        "FASTA download",
      );
    } finally {
      win.URL.createObjectURL = create;
      win.HTMLAnchorElement.prototype.click = click;
    }
    $("mode").value = "motif";
    $("mode").dispatchEvent(new Event("change"));
    $("motif").value = "##ACDE##";
    $("precision").value = "fp16";
    $("run").click();
    await until(() => $("status").textContent.startsWith("Done."));
    assert(
      $("sequence").textContent.slice(2, 6) === "ACDE",
      "Demo motif not preserved",
    );
    $("mode").value = "unconditional";
    $("length").value = "24";
    $("run").click();
    await until(() => $("status").textContent.startsWith("Generating"));
    $("cancel").click();
    await until(() => $("status").textContent === "Cancelled.");
    $("release").click();
    assert($("status").textContent === "GPU buffers released.", "Demo release");
    return {
      unconditional: true,
      motif: true,
      precisionSelector: true,
      progress: true,
      cancel: true,
      fastaDownload: true,
      releaseGPU: true,
    };
  } finally {
    frame.remove();
  }
}
