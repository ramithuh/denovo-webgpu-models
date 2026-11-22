import {
  createSidProteinEngine,
  createSidProteinWorker,
} from "../webgpu/index.js";
import { loadWeights, decodeTensor, diffStats } from "../shared/index.js";
const assert = (v, m) => {
  if (!v) throw new Error(m);
};
async function rejects(fn, match) {
  try {
    await fn();
  } catch (e) {
    if (match && !match.test(e.message + " " + e.name)) throw e;
    return;
  }
  throw new Error("Expected failure");
}
export async function lifecycle(engine, c, read) {
  const x = await read(c.forwardInput),
    baseline = engine.memoryUsage().totalBytes;
  await rejects(
    () =>
      engine.forward(x, 0.37, {
        onTrace() {
          throw new Error("injected tracing failure");
        },
      }),
    /injected/,
  );
  assert(
    engine.memoryUsage().transientBuffers === 0 &&
      engine.memoryUsage().totalBytes === baseline,
    "Error path leaked buffers",
  );
  const ac = new AbortController();
  await rejects(
    () =>
      engine.forward(x, 0.37, {
        signal: ac.signal,
        onTrace() {
          ac.abort();
        },
      }),
    /Abort/,
  );
  assert(
    engine.memoryUsage().transientBuffers === 0 &&
      engine.memoryUsage().totalBytes === baseline,
    "Abort leaked buffers",
  );
  await rejects(() => engine.forward(x, NaN), /time/);
  await rejects(() => engine.generate({ length: 0 }), /length/);
  const out = await engine.forward(x, 0.37);
  assert(
    diffStats(out, await read(c.forward)).relRms < 1e-4,
    "Recovery after abort failed",
  );
  return {
    errorCleanup: true,
    abortCleanup: true,
    recovery: true,
    transientBuffers: 0,
  };
}
export async function cacheContracts() {
  const base = new URL("/.cache/loader", location.href).href,
    read = async (url) => {
      const r = await fetch(url);
      if (!r.ok) throw new Error("Fixture missing");
      return r.arrayBuffer();
    };
  const expected = [1, 2, 3, -2, -1, 0, 1.5, 0, -1.5, 7, 8, 9];
  for (const cache of ["cache", "opfs"]) {
    const a = await loadWeights(base, { cache });
    assert(
      JSON.stringify([...decodeTensor(a.weights["test.weight"])]) ===
        JSON.stringify(expected),
      "Multipart loader mismatch",
    );
    const original = globalThis.fetch;
    globalThis.fetch = () => {
      throw new Error("Warm cache attempted network");
    };
    try {
      const b = await loadWeights(base, { cache });
      assert(b.manifest.precision === "int8", "Warm cache failed");
    } finally {
      globalThis.fetch = original;
    }
  }
  for (const cache of ["cache", "opfs"]) {
    const target = cache === "cache" ? globalThis : navigator,
      key = cache === "cache" ? "caches" : "storage",
      previous = Object.getOwnPropertyDescriptor(target, key);
    let warnings = 0;
    Object.defineProperty(target, key, {
      configurable: true,
      value:
        cache === "cache"
          ? {
              open: async () => {
                throw new DOMException(
                  "Private storage denied",
                  "SecurityError",
                );
              },
            }
          : {
              getDirectory: async () => {
                throw new DOMException(
                  "Private storage denied",
                  "SecurityError",
                );
              },
            },
    });
    try {
      await loadWeights(base, {
        cache,
        onProgress: (e) => {
          if (e.phase === "cache-warning") warnings++;
        },
      });
      assert(warnings > 0, "Private-mode fallback warning missing");
    } finally {
      if (previous) Object.defineProperty(target, key, previous);
      else delete target[key];
    }
  }
  const b = await loadWeights(base, { read });
  assert(b.weights["test.weight"].bytes.length === 12, "Injected read failed");
  await rejects(
    () =>
      loadWeights(base, {
        read: async (url) => {
          const data = await read(url);
          if (url.endsWith(".safetensors"))
            new Uint8Array(data)[data.byteLength - 1] ^= 1;
          return data;
        },
      }),
    /checksum/,
  );
  return {
    cacheApiColdWarm: true,
    opfsColdWarm: true,
    safariPrivateModeSimulated: true,
    sha256Rejection: true,
    multipartInt8ScaleColocation: true,
  };
}
export async function externalContracts(c, read) {
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  assert(adapter, "No hardware adapter");
  const device = await adapter.requestDevice(),
    live = new Set();
  let destroyedDevice = 0;
  const proxy = new Proxy(device, {
    get(target, key) {
      if (key === "adapterInfo") return adapter.info;
      if (key === "createBuffer")
        return (options) => {
          const b = target.createBuffer(options),
            destroy = b.destroy.bind(b);
          live.add(b);
          b.destroy = () => {
            live.delete(b);
            destroy();
          };
          return b;
        };
      if (key === "destroy")
        return () => {
          destroyedDevice++;
          target.destroy();
        };
      const v = Reflect.get(target, key, target);
      return typeof v === "function" ? v.bind(target) : v;
    },
  });
  try {
    const limited = new Proxy(proxy, {
      get(target, key) {
        if (key === "limits")
          return { maxBufferSize: 64, maxStorageBufferBindingSize: 64 };
        return Reflect.get(target, key);
      },
    });
    await rejects(
      () =>
        createSidProteinEngine({
          device: limited,
          baseUrl: "/weights/int8",
          precision: "int8",
          allowSoftwareFallback: false,
        }),
      /maxBufferSize/,
    );
    assert(live.size === 0, "Constructor failure leaked buffers");
    let shards = 0;
    await rejects(
      () =>
        createSidProteinEngine({
          device: proxy,
          baseUrl: "/weights/int8",
          precision: "int8",
          allowSoftwareFallback: false,
          read: async (url) => {
            const r = await fetch(url);
            if (!r.ok) throw new Error("Weights missing");
            const d = await r.arrayBuffer();
            if (url.endsWith(".safetensors") && ++shards === 2)
              new Uint8Array(d)[d.byteLength - 1] ^= 1;
            return d;
          },
        }),
      /checksum/,
    );
    assert(live.size === 0, "Failed load leaked GPU buffers");
    const engine = await createSidProteinEngine({
      device: proxy,
      baseUrl: "/weights/int8",
      precision: "int8",
      allowSoftwareFallback: false,
      useF16: false,
    });
    try {
      const actual = await engine.forward(await read(c.forwardInput), 0.37),
        expected = await read(c.forward);
      assert(
        diffStats(actual, expected).relRms < 1e-4,
        "External GPUDevice parity failed",
      );
    } finally {
      engine.destroy();
      engine.destroy();
    }
    assert(
      live.size === 0 && destroyedDevice === 0,
      "External device ownership/leak contract failed",
    );
    // Verify the borrowed device is still usable after engine destruction.
    const buf = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.COPY_DST,
    });
    buf.destroy();
  } finally {
    device.destroy();
  }
  return {
    externalDevice: true,
    idempotentDestroy: true,
    loadFailureCleanup: true,
    maxBufferSizeRejection: true,
    liveBuffers: 0,
  };
}
export async function workerContracts(c, read) {
  const client = createSidProteinWorker();
  try {
    const meta = await client.call("init", {
      baseUrl: new URL("/weights/int8", location.href).href,
      precision: "int8",
      allowSoftwareFallback: false,
    });
    assert(!meta.isFallbackAdapter, "Worker software adapter");
    const x = await read(c.forwardInput),
      out = await client.call("forward", { coordinates: x, time: 0.37 });
    assert(
      diffStats(out, await read(c.forward)).relRms < 1e-4,
      "Worker forward parity failed",
    );
    const ac = new AbortController();
    await rejects(
      () =>
        client.call(
          "generate",
          {
            length: c.length,
            steps: c.steps,
            noiseScale: c.noiseScale,
            noise: undefined,
          },
          {
            signal: ac.signal,
            onProgress: (kind) => {
              if (kind === "step") ac.abort();
            },
          },
        ),
      /Abort/,
    );
    const memory = await client.call("memoryUsage");
    assert(memory.transientBuffers === 0, "Worker abort leaked");
    const result = await client.call("generate", {
        length: c.length,
        steps: c.steps,
        noiseScale: c.noiseScale,
        noise: await read(c.noise),
      }),
      ref = await read(c.trajectory);
    assert(
      diffStats(result.coordinates, ref.slice(-c.length * 3)).relRms < 1e-4,
      "Worker sampling parity failed",
    );
    const end = await client.call("destroy");
    assert(end.totalBytes === 0, "Worker destroy leaked");
    return {
      forward: true,
      sampling16: true,
      abortCleanup: true,
      afterDestroyBytes: end.totalBytes,
      adapterInfo: meta.adapterInfo,
    };
  } finally {
    await client.destroy();
  }
}
export async function portableHalfContract(c, read) {
  const engine = await createSidProteinEngine({
    baseUrl: "/weights/fp16",
    precision: "fp16",
    useF16: false,
    allowSoftwareFallback: false,
  });
  try {
    assert(!engine.useF16, "Portable half path enabled native f16");
    const out = await engine.forward(await read(c.forwardInput), 0.37);
    assert(
      diffStats(out, await read(c.forward)).relRms < 1e-4,
      "Portable half forward mismatch",
    );
    const generated = await engine.generate({
      length: c.length,
      steps: c.steps,
      noiseScale: c.noiseScale,
      noise: await read(c.noise),
    });
    assert(
      diffStats(
        generated.coordinates,
        (await read(c.trajectory)).slice(-c.length * 3),
      ).relRms < 1e-4,
      "Portable half sampler mismatch",
    );
    return { packedHalfWithoutShaderF16: true, sampling16: true };
  } finally {
    engine.destroy();
    assert(
      engine.memoryUsage().totalBytes === 0,
      "Portable half destroy leaked",
    );
  }
}
export async function sensitivityRegression(engine, read) {
  const r = await fetch("/fixtures/sensitivity_regression.json");
  if (!r.ok) throw new Error("Missing sensitivity regression");
  const c = await r.json();
  const generated = await engine.generate({
      length: c.length,
      steps: c.steps,
      noiseScale: c.noiseScale,
      noise: await read(c.noise),
    }),
    ref = await read(c.trajectory),
    sampling = diffStats(generated.coordinates, ref.slice(-c.length * 3)),
    trajectory = diffStats(
      Float32Array.from(generated.trajectory.flatMap((a) => Array.from(a))),
      ref,
    );
  assert(
    sampling.relRms < 1e-4 && trajectory.relRms < 1e-4,
    "Original sensitivity regression failed " +
      JSON.stringify({ sampling, trajectory }),
  );
  return { name: c.name, seed: c.seed, sampling, trajectory };
}
