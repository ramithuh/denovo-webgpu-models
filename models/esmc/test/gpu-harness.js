import { createEsmcEngine, requestWebGpuDevice } from "../webgpu/index.js";
import { oracle, verifyProtein, verifyTokens, check } from "./parity.js";
const log = (s) => fetch("/log", { method: "POST", body: s });
async function fixture(precision, name) {
  const base = `/fixtures/oracle_${precision}_${name}`;
  const meta = await (await fetch(base + ".json")).json();
  return {
    meta,
    ref: oracle(meta, await (await fetch(base + ".bin")).arrayBuffer()),
  };
}
export async function run() {
  const params = new URL(import.meta.url).searchParams,
    precision = params.get("precision") ?? "fp32",
    useF16 = params.get("native") === "1";
  const { adapter, device, hasF16 } = await requestWebGpuDevice({ useF16 });
  const i = adapter.info,
    adapterInfo = {
      vendor: i.vendor,
      architecture: i.architecture,
      device: i.device,
      description: i.description,
      isFallbackAdapter: i.isFallbackAdapter,
    };
  check(
    !i.isFallbackAdapter &&
      !/swiftshader|llvmpipe|lavapipe|software/i.test(
        JSON.stringify(adapterInfo),
      ),
    "Real hardware required",
  );
  if (useF16) check(hasF16, "shader-f16 unavailable");
  const errors = [];
  device.addEventListener("uncapturederror", (e) =>
    errors.push(e.error.message),
  );
  await log(JSON.stringify(adapterInfo));
  const start = performance.now(),
    engine = await createEsmcEngine({
      baseUrl: "/weights/esmc-300m",
      precision,
      device,
      cache: "none",
      useF16,
      onProgress: (p) => {
        if (p.loaded === p.total)
          log(`${precision} weights loaded: ${p.loaded} bytes`);
      },
    }),
    loadMs = performance.now() - start,
    metrics = {};
  const baseline = engine.memoryUsage();
  if (useF16) check(engine.useF16, "Native F16 path was not enabled");
  const features = [...device.features];
  try {
    for (const name of ["trp_cage", "ubiquitin"]) {
      await log(`${precision} ${name}`);
      const { meta, ref } = await fixture(precision, name);
      metrics[name] = await verifyProtein(engine, meta, ref, {
        log: (s) => log(`${precision} ${name} ${s}`),
        warm: true,
      });
      check(
        engine.memoryUsage().transientBuffers === 0 &&
          engine.memoryUsage().totalBytes === baseline.totalBytes,
        "Inference leaked buffers",
      );
    }
    const { meta, ref } = await fixture(precision, "tokens");
    metrics.tokens = await verifyTokens(engine, meta, ref);
    const n = await engine.run("AC", { mode: "naturalness" }),
      m = await engine.run("AC", { mode: "masked" });
    check(
      n.pseudoPerplexity === m.pseudoPerplexity,
      "Naturalness route differs",
    );
    const c = new AbortController();
    setTimeout(() => c.abort(), 10);
    let aborted = false;
    try {
      await engine.forward("NLYIQWLKDGGPSSGRPPPS", { signal: c.signal });
    } catch (e) {
      check(e.name === "AbortError", e.message);
      aborted = true;
    }
    check(aborted, "In-flight cancellation ignored");
    check(
      engine.memoryUsage().transientBuffers === 0,
      "Cancellation leaked buffers",
    );
    const original = engine.pipeline;
    let calls = 0;
    engine.pipeline = function (...args) {
      if (++calls === 4) throw new Error("injected pipeline failure");
      return original.apply(this, args);
    };
    try {
      await engine.forward("AC");
      throw new Error("Expected injected failure");
    } catch (e) {
      check(e.message === "injected pipeline failure", e.message);
    } finally {
      engine.pipeline = original;
    }
    check(
      engine.memoryUsage().transientBuffers === 0 &&
        engine.memoryUsage().totalBytes === baseline.totalBytes,
      "Exception leaked buffers",
    );
    try {
      engine.buffer(device.limits.maxStorageBufferBindingSize + 4);
      throw new Error("Oversize buffer admitted");
    } catch (e) {
      check(e instanceof RangeError, e.message);
    }
    try {
      await engine.forward("ac");
      throw new Error("Invalid sequence admitted");
    } catch (e) {
      check(e instanceof TypeError, e.message);
    }
  } finally {
    engine.destroy();
    engine.destroy();
    check(engine.memoryUsage().totalBytes === 0, "Destroy leaked");
  }
  // The externally supplied device must still work after engine destruction.
  const b = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(b, 0, new Uint32Array([1]));
  await device.queue.onSubmittedWorkDone();
  b.destroy();
  check(errors.length === 0, errors.join("\n"));
  device.destroy();
  return {
    adapterInfo,
    precision,
    useF16,
    loadMs,
    weightBytes: baseline.totalBytes,
    metrics,
    validationErrors: errors,
    features,
  };
}
