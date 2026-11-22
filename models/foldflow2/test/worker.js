export async function workerTest(v) {
  const worker = new Worker("/webgpu/worker.js", { type: "module" }),
    pending = new Map();
  let id = 0,
    abortOnProgress = false;
  worker.onmessage = ({ data }) => {
    if (data.progress) {
      if (abortOnProgress) {
        abortOnProgress = false;
        worker.postMessage({ type: "abort" });
      }
      return;
    }
    const p = pending.get(data.id);
    if (p) {
      pending.delete(data.id);
      data.error ? p.reject(Error(data.error.message)) : p.resolve(data.result);
    }
  };
  const call = (type, options) =>
    new Promise((resolve, reject) => {
      const n = ++id;
      pending.set(n, { resolve, reject });
      worker.postMessage({ id: n, type, options });
    });
  try {
    abortOnProgress = true;
    let rejected = false;
    try {
      await call("init", {
        baseUrl: "/weights/int8",
        allowSoftwareFallback: false,
      });
    } catch (e) {
      if (!/abort/i.test(e.message)) throw e;
      rejected = true;
    }
    if (!rejected) throw Error("Worker initialization did not abort");
    const afterAbort = await call("memory");
    if (afterAbort.bytes !== 0) throw Error("Worker initialization abort leak");
    const info = await call("init", {
      baseUrl: "/weights/int8",
      allowSoftwareFallback: false,
    });
    const out = await call("sample", {
      length: 32,
      steps: 2,
      initial: v.initial,
    });
    const memory = await call("memory");
    if (memory.transientBuffers !== 0) throw Error("Worker transient leak");
    await call("destroy");
    const destroyed = await call("destroy");
    if (destroyed.bytes !== 0) throw Error("Worker destroy leak");
    return { out, info, memory, destroyed, initializationAbort: true };
  } finally {
    worker.terminate();
  }
}
