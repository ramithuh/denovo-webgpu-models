import { createProteinaEngine } from "../webgpu/index.js";
import { assert, parity } from "./parity.js";
async function rejects(fn, label) {
  let threw = false;
  try {
    await fn();
  } catch {
    threw = true;
  }
  assert(threw, label ?? "Expected rejection");
}
export async function hygiene(engine, device, ref) {
  const c = ref.meta.denoise[0],
    x = ref.tensor(c.x),
    options = { t: c.t };
  await rejects(() => engine.forward([NaN, 0, 0]));
  await rejects(() => engine.forward(x, { t: 1 }));
  await rejects(() => engine.forward(x, { fold: [-1, 0, 0] }));
  await rejects(() => engine.forward(x, { xSC: [0] }));
  const originalSubmit = device.queue.submit;
  device.queue.submit = () => {
    throw Error("Injected submit failure");
  };
  try {
    await rejects(() => engine.forward(x, options));
  } finally {
    device.queue.submit = originalSubmit;
  }
  assert(engine.memoryUsage().transientBuffers === 0, "Error transient leak");
  const create = device.createBuffer;
  let count = 0;
  // Use a fresh length, so construction must allocate input scratch buffers.
  device.createBuffer = function (...args) {
    if (++count === 2) throw Error("Injected input allocation error");
    return create.apply(this, args);
  };
  try {
    await rejects(() => engine.forward(new Float32Array(111), options));
  } finally {
    device.createBuffer = create;
  }
  assert(
    engine.memoryUsage().transientBuffers === 0,
    "Constructor error transient leak",
  );
  await rejects(() =>
    engine.forward(
      new Float32Array(
        3 *
          Math.ceil(
            Math.sqrt(device.limits.maxStorageBufferBindingSize / (512 * 4)) +
              10,
          ),
      ),
      options,
    ),
  );
  const before = new AbortController();
  before.abort();
  await rejects(() => engine.forward(x, { ...options, signal: before.signal }));
  const during = new AbortController(),
    promise = engine.forward(x, { ...options, signal: during.signal });
  setTimeout(() => during.abort(), 0);
  await rejects(() => promise);
  assert(
    engine.memoryUsage().transientBuffers === 0,
    "Live abort transient leak",
  );
  const sampleAbort = new AbortController();
  await rejects(() =>
    engine.sample({
      length: 32,
      steps: 3,
      signal: sampleAbort.signal,
      onProgress: () => sampleAbort.abort(),
    }),
  );
  assert(engine.memoryUsage().transientBuffers === 0, "Sampler abort leak");
  await rejects(() =>
    engine.sample({
      length: 32,
      steps: 3,
      onProgress() {
        throw Error("Injected progress error");
      },
    }),
  );
  assert(engine.memoryUsage().transientBuffers === 0, "Callback error leak");
  const a = await engine.forward(x, options),
    b = await engine.forward(x, options);
  parity(a.coordinates, ref.tensor(c.output), "Recovery");
  parity(b.coordinates, a.coordinates, "Deterministic replay");
  return {
    invalidInput: true,
    submitError: true,
    allocationError: true,
    bufferLimit: true,
    preAborted: true,
    liveAbort: true,
    samplerAbort: true,
    callbackError: true,
    recovery: true,
    transientBuffers: 0,
  };
}
export async function initializationCleanup(device, baseUrl) {
  const create = device.createBuffer,
    allocated = new Set();
  device.createBuffer = function (...a) {
    const b = create.apply(this, a),
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
      createProteinaEngine({
        baseUrl,
        precision: "int8",
        device,
        allowSoftwareFallback: false,
        cache: "none",
        signal: abort.signal,
        onProgress: (p) => {
          if (p.phase === "weights") abort.abort();
        },
      }),
    );
    assert(allocated.size === 0, "Initialization abort leak");
    await rejects(() =>
      createProteinaEngine({
        baseUrl,
        precision: "int8",
        device,
        allowSoftwareFallback: false,
        cache: "none",
        onProgress() {
          throw Error("Injected load callback error");
        },
      }),
    );
    assert(allocated.size === 0, "Initialization error leak");
    await rejects(() =>
      createProteinaEngine({
        baseUrl: baseUrl + "/missing",
        precision: "int8",
        device,
        allowSoftwareFallback: false,
        cache: "none",
      }),
    );
    assert(allocated.size === 0, "Missing weights leak");
    return { abortBuffers: 0, errorBuffers: 0, missingWeightsBuffers: 0 };
  } finally {
    device.createBuffer = create;
    for (const b of allocated) b.destroy();
  }
}
