import { createProteinaEngine, rejectSoftware } from "../webgpu/index.js";
import { oracle, checkEngine, assert, parity } from "./parity.js";
import { hygiene, initializationCleanup } from "./hygiene.js";
import { demoTest } from "./demo.js";
import { workerTests } from "./worker-tests.js";
const read = async (url) => {
  const r = await fetch("/" + url);
  if (!r.ok) throw Error("Missing fixture " + url);
  return r.arrayBuffer();
};
const log = (message) => fetch("/log", { method: "POST", body: message });
export async function run() {
  assert(navigator.gpu, "Hardware WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  assert(adapter, "Hardware adapter missing");
  rejectSoftware(adapter.info, adapter.isFallbackAdapter);
  assert(
    adapter.info.vendor === "nvidia",
    "Expected NVIDIA verification hardware",
  );
  assert(adapter.features.has("shader-f16"), "Native f16 unavailable");
  const device = await adapter.requestDevice({
    requiredFeatures: ["shader-f16"],
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  });
  const results = {
      skipped: 0,
      precisions: {},
      userAgent: navigator.userAgent,
    },
    baseUrl = location.origin + "/weights",
    refs = {};
  try {
    results.initialization = await initializationCleanup(device, baseUrl);
    for (const precision of ["fp32", "fp16", "int8"]) {
      const ref = (refs[precision] = await oracle(precision, read)),
        start = performance.now();
      const engine = await createProteinaEngine({
        baseUrl,
        precision,
        device,
        cache: "none",
        allowSoftwareFallback: false,
      });
      results.adapter = engine.adapterInfo;
      try {
        results.precisions[precision] = {
          loadMs: performance.now() - start,
          ...(await checkEngine(engine, ref, log, true)),
          memory: engine.memoryUsage(),
        };
        assert(engine.memoryUsage().transientBuffers === 0, "Transient leak");
        const timing = {};
        for (const L of [32, 64]) {
          const c = ref.meta.denoise.find((c) => c.length === L),
            x = ref.tensor(c.x),
            runs = [];
          for (let i = 0; i < 3; i++)
            runs.push((await engine.forward(x, { t: c.t })).timingMs);
          timing[L] = {
            runsMs: runs,
            medianMs: runs.slice().sort((a, b) => a - b)[1],
          };
        }
        results.precisions[precision].warmTimings = timing;
        if (precision === "int8") {
          await log("Error and abort cleanup");
          results.hygiene = await hygiene(engine, device, ref);
        }
      } finally {
        engine.destroy();
        engine.destroy();
        assert(engine.memoryUsage().totalBytes === 0, "Destroy leak");
        results.precisions[precision].afterDestroyBytes =
          engine.memoryUsage().totalBytes;
      }
    }
    const bytes = (p) => results.precisions[p].memory.weightBytes;
    results.memoryRatios = {
      fp16: bytes("fp16") / bytes("fp32"),
      int8: bytes("int8") / bytes("fp32"),
    };
    assert(
      results.memoryRatios.fp16 < 0.51 && results.memoryRatios.int8 < 0.26,
      "Weights must remain packed in VRAM",
    );
    results.quantizationDrift = {};
    for (const p of ["fp16", "int8"]) {
      results.quantizationDrift[p] = { denoise: [], samples: [] };
      for (const kind of ["denoise", "samples"])
        for (let i = 0; i < refs[p].meta[kind].length; i++) {
          const c = refs[p].meta[kind][i],
            baseline = refs.fp32.meta[kind][i];
          results.quantizationDrift[p][kind].push({
            id: c.id,
            ...parity(
              refs[p].tensor(c.output),
              refs.fp32.tensor(baseline.output),
              "Quantization drift",
              Infinity,
            ),
          });
        }
    }
    const native = await createProteinaEngine({
      baseUrl,
      precision: "fp16",
      useF16: true,
      device,
      cache: "none",
      allowSoftwareFallback: false,
    });
    try {
      const c = refs.fp16.meta.denoise[0];
      results.nativeF16 = parity(
        (await native.forward(refs.fp16.tensor(c.x), { t: c.t })).coordinates,
        refs.fp16.tensor(c.output),
        "Native packed f16",
      );
    } finally {
      native.destroy();
      assert(native.memoryUsage().totalBytes === 0, "Native destroy leak");
    }
  } finally {
    device.destroy();
  }
  await log("Worker Cache API / OPFS / Private Mode");
  results.workerStorage = await workerTests(refs.int8);
  await log("Static demo");
  results.demo = await demoTest();
  return results;
}
