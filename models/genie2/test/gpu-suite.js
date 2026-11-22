import { createGenie2Engine } from "../webgpu/index.js";
import { acquireDevice } from "../webgpu/device.js";
import { GPUAlgebra } from "../webgpu/algebra.js";
import { createGenie2Worker } from "../webgpu/worker-client.js";
import { diffStats } from "../shared/weights.js";
import { loadFixture, log } from "./browser-fixture.js";
const assert = (ok, message) => {
  if (!ok) throw Error(message);
};
export async function run() {
  const records = [],
    memory = {},
    lifecycle = [],
    timings = {};
  let adapterInfo, workerResult;
  for (const precision of ["fp32", "fp16", "int8"]) {
    const started = performance.now(),
      e = await createGenie2Engine({
        baseUrl: `/weights/${precision}`,
        precision,
        allowSoftwareFallback: false,
      }),
      loadMs = performance.now() - started;
    adapterInfo = e.adapterInfo;
    memory[precision] = e.memoryUsage();
    await log({ precision, adapterInfo, loadMs });
    try {
      const f0 = await loadFixture(precision, 0),
        args0 = {
          coordinates: f0.get("initial"),
          features: f0.features,
          timestep: 2,
          steps: 2,
        };
      const coldStart = performance.now(),
        cold = await e.forward(args0),
        coldMs = performance.now() - coldStart;
      const warmStart = performance.now(),
        warm = await e.forward(args0),
        warmMs = performance.now() - warmStart;
      assert(
        diffStats(cold.z, warm.z).relRms < 1e-6,
        "Cold/warm output changed",
      );
      timings[precision] = {
        loadMs,
        coldForwardMs: coldMs,
        warmForwardMs: warmMs,
        length: 24,
      };
      for (let i = 0; i < 6; i++) {
        const { c, get, features } = await loadFixture(precision, i),
          start = performance.now();
        const out = await e.sample({
          length: c.length,
          features,
          steps: c.steps,
          scale: c.scale,
          initialNoise: get("initial"),
          noises: Array.from({ length: c.steps }, (_, k) =>
            get(`step${c.steps - k}.noise`),
          ),
          trajectory: true,
        });
        const row = {
          case: c.name,
          final: diffStats(out.coordinates, get("final")),
          trajectory: out.trajectory.map((x, k) =>
            diffStats(x, get(`step${c.steps - k}.output`)),
          ),
          elapsedMs: performance.now() - start,
          loadMs,
        };
        records.push(row);
        await log(row);
        assert(e.memoryUsage().transientBuffers === 0, "Transient leak");
        if (i === 0) {
          const opts = {
            coordinates: get("initial"),
            features,
            timestep: 2,
            steps: 2,
          };
          for (const kind of ["abort", "error"]) {
            const control = new AbortController();
            let caught = false;
            try {
              await e.forward({
                ...opts,
                signal: control.signal,
                onStage: () => {
                  if (kind === "abort") control.abort();
                  else throw Error("injected");
                },
              });
            } catch (err) {
              assert(
                kind === "abort"
                  ? err.name === "AbortError"
                  : err.message === "injected",
                "Unexpected lifecycle exception " + err,
              );
              caught = true;
            }
            assert(
              caught && e.memoryUsage().transientBuffers === 0,
              "Error/abort cleanup failure",
            );
            lifecycle.push({ precision, kind, transientBuffers: 0 });
          }
          if (precision === "fp32") {
            const first = performance.now(),
              a = await e.forward(opts),
              coldMs = performance.now() - first,
              second = performance.now(),
              b = await e.forward(opts),
              warmMs = performance.now() - second;
            assert(
              diffStats(a.z, b.z).relRms < 1e-6,
              "Warm output not repeatable",
            );
            row.repeatTiming = {
              firstMeasuredMs: coldMs,
              repeatMs: warmMs,
              prewarmed: true,
            };
            const worker = createGenie2Worker();
            try {
              const info = await worker.init({
                baseUrl: new URL("/weights/fp32", location.href).href,
                precision,
                allowSoftwareFallback: false,
              });
              const w = await worker.forward(opts);
              workerResult = {
                ...info,
                vsMain: diffStats(w.z, b.z),
                memory: await worker.memoryUsage(),
              };
              assert(workerResult.vsMain.relRms < 1e-6, "Worker mismatch");
            } finally {
              await worker.destroy();
              await worker.destroy();
            }
          }
        }
      }
    } finally {
      e.destroy();
      e.destroy();
      assert(e.memoryUsage().bytes === 0, "Destroy leak");
    }
  }
  const acquired = await acquireDevice({
    allowSoftwareFallback: false,
    useF16: true,
  });
  try {
    const owned = await createGenie2Engine({
      baseUrl: "/weights/int8",
      precision: "int8",
      device: acquired.device,
      allowSoftwareFallback: false,
      useF16: true,
    });
    owned.destroy();
    const probe = acquired.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.COPY_DST,
    });
    assert(probe.size === 4, "External device was destroyed");
    probe.destroy();
    const ops = new GPUAlgebra(acquired.device);
    let limited = false;
    try {
      ops.alloc([
        Math.floor(
          Math.min(
            acquired.device.limits.maxBufferSize,
            acquired.device.limits.maxStorageBufferBindingSize,
          ) / 4,
        ) + 1,
      ]);
    } catch (e) {
      limited = /exceeds/.test(e.message);
    } finally {
      ops.destroy();
    }
    assert(limited, "Missing buffer limit guard");
    lifecycle.push({
      externalDeviceSurvived: true,
      bufferLimitsEnforced: true,
    });
  } finally {
    acquired.device.destroy();
  }
  const ratio16 = memory.fp16.bytes / memory.fp32.bytes,
    ratio8 = memory.int8.bytes / memory.fp32.bytes;
  assert(
    ratio16 < 0.52 && ratio16 > 0.49 && ratio8 < 0.27 && ratio8 > 0.24,
    "Weights were expanded in GPU memory",
  );
  const storage = [];
  for (const cache of ["cache", "opfs"]) {
    const options = {
      baseUrl: "/weights/int8",
      precision: "int8",
      allowSoftwareFallback: false,
      cache,
    };
    const first = await createGenie2Engine(options);
    first.destroy();
    const realFetch = globalThis.fetch;
    try {
      globalThis.fetch = async (url) => {
        if (String(url).includes("/weights/"))
          throw Error("Offline cache miss");
        return realFetch(url);
      };
      const second = await createGenie2Engine(options);
      second.destroy();
      storage.push({ cache, offlineReload: true });
    } finally {
      globalThis.fetch = realFetch;
    }
  }
  const original = navigator.storage.getDirectory.bind(navigator.storage);
  let warning = false;
  try {
    navigator.storage.getDirectory = async () => {
      throw new DOMException("Private storage denied", "SecurityError");
    };
    const e = await createGenie2Engine({
      baseUrl: "/weights/int8",
      precision: "int8",
      cache: "opfs",
      allowSoftwareFallback: false,
      onProgress: (p) => {
        if (p.phase === "cache-warning") warning = true;
      },
    });
    e.destroy();
    assert(warning, "Denied-storage fallback not exercised");
    storage.push({ cache: "opfs", simulatedPrivateModeFallback: true });
  } finally {
    navigator.storage.getDirectory = original;
  }
  return {
    reference: "unmodified pinned PyTorch",
    threshold: 1e-4,
    adapterInfo,
    records,
    memory,
    timings,
    ratios: { fp16: ratio16, int8: ratio8 },
    lifecycle,
    workerResult,
    storage,
    skipped: 0,
    passed: records.every((r) => r.final.relRms < 1e-4),
  };
}
