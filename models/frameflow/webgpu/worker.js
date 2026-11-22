/** Dedicated Worker client; callbacks and AbortSignals stay on the caller's thread. */
export async function createFrameFlowWorker(options = {}) {
  const {
    workerUrl = new URL("./worker-entry.js", import.meta.url),
    signal,
    onProgress,
    ...settings
  } = options;
  const worker = new Worker(workerUrl, { type: "module" }),
    pending = new Map();
  let id = 0,
    closed = false;
  worker.onmessage = ({ data }) => {
    const p = pending.get(data.id);
    if (!p) return;
    if (data.progress) {
      p.onProgress?.(data.progress);
      return;
    }
    pending.delete(data.id);
    p.cleanup();
    if (data.error) {
      const e = new Error(data.error.message);
      e.name = data.error.name;
      p.reject(e);
    } else p.resolve(data.result);
  };
  worker.onerror = (e) => {
    for (const p of pending.values()) {
      p.cleanup();
      p.reject(Error(e.message));
    }
    pending.clear();
  };
  function call(method, args, options = {}) {
    if (closed) return Promise.reject(Error("Worker destroyed"));
    options.signal?.throwIfAborted();
    const n = ++id;
    return new Promise((resolve, reject) => {
      const abort = () => worker.postMessage({ method: "abort", target: n });
      const cleanup = () => options.signal?.removeEventListener("abort", abort);
      pending.set(n, {
        resolve,
        reject,
        cleanup,
        onProgress: options.onProgress,
      });
      options.signal?.addEventListener("abort", abort, { once: true });
      worker.postMessage({ id: n, method, args });
    });
  }
  try {
    const meta = await call(
      "init",
      {
        ...settings,
        baseUrl: new URL(settings.baseUrl, globalThis.location.href).href,
      },
      { signal, onProgress },
    );
    return {
      ...meta,
      forward(input, o) {
        return call("forward", input, o);
      },
      sample(options) {
        const { signal, onProgress, ...args } = options;
        return call("sample", args, { signal, onProgress });
      },
      memoryUsage() {
        return call("memory");
      },
      async destroy() {
        if (closed) return;
        try {
          await call("destroy");
        } finally {
          closed = true;
          worker.terminate();
          for (const p of pending.values()) {
            p.cleanup();
            p.reject(Error("Worker destroyed"));
          }
          pending.clear();
        }
      },
    };
  } catch (e) {
    closed = true;
    worker.terminate();
    throw e;
  }
}
