import { assert, parity } from "./parity.js";
export async function workerTests(ref) {
  const worker = new Worker(new URL("./worker.js", import.meta.url), {
      type: "module",
    }),
    waiting = new Map();
  let seq = 0,
    readyResolve;
  const ready = new Promise((r) => (readyResolve = r));
  worker.onmessage = ({ data }) => {
    if (data.ready) {
      readyResolve();
      return;
    }
    if (data.progress) return;
    const p = waiting.get(data.id);
    if (!p) return;
    waiting.delete(data.id);
    if (data.error) p.reject(Error(data.error));
    else p.resolve(data.result);
  };
  worker.onerror = (e) => {
    for (const p of waiting.values()) p.reject(Error(e.message));
    waiting.clear();
  };
  const call = (cmd, args) =>
    new Promise((resolve, reject) => {
      const id = ++seq;
      waiting.set(id, { resolve, reject });
      worker.postMessage({ id, cmd, args });
    });
  const c = ref.meta.denoise[0],
    result = {},
    baseUrl = location.origin + "/weights";
  const forward = async () => {
    const out = await call("forward", { coordinates: ref.tensor(c.x), t: c.t });
    return parity(out.coordinates, ref.tensor(c.output), "Worker");
  };
  await ready;
  try {
    for (const cache of ["cache", "opfs"]) {
      await call("init", {
        baseUrl,
        precision: "int8",
        cache,
        allowSoftwareFallback: false,
      });
      await forward();
      await call("destroy");
      await call("test-storage", { offline: true });
      await call("init", {
        baseUrl,
        precision: "int8",
        cache,
        allowSoftwareFallback: false,
      });
      result[cache] = { offline: true, ...(await forward()) };
      assert(
        (await call("memory")).transientBuffers === 0,
        "Worker transient leak",
      );
      assert((await call("destroy")).totalBytes === 0, "Worker destroy leak");
      await call("test-storage", { offline: false });
      await call("test-storage", { fail: cache });
      await call("init", {
        baseUrl,
        precision: "int8",
        cache,
        allowSoftwareFallback: false,
      });
      result[cache + "PrivateModeFallback"] = await forward();
      await call("destroy");
      await call("test-storage", {});
    }
  } finally {
    worker.terminate();
  }
  return result;
}
