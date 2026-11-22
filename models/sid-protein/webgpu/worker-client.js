/** All work and weights remain in the dedicated worker. */
export function createSidProteinWorker({
  worker = new Worker(new URL("./worker.js", import.meta.url), {
    type: "module",
  }),
} = {}) {
  let serial = 0,
    closed = false;
  const pending = new Map();
  worker.onmessage = ({ data }) => {
    const p = pending.get(data.id);
    if (!p) return;
    if (data.event) {
      p.onProgress?.(data.event, data.value);
      return;
    }
    pending.delete(data.id);
    p.cleanup();
    if (data.error) {
      const e = new Error(data.error.message);
      e.name = data.error.name;
      p.reject(e);
    } else p.resolve(data.value);
  };
  worker.onerror = (e) => {
    for (const p of pending.values()) {
      p.cleanup();
      p.reject(new Error(e.message));
    }
    pending.clear();
  };
  const call = (method, args, { signal, onProgress } = {}) => {
    if (closed) return Promise.reject(new Error("Worker terminated"));
    signal?.throwIfAborted();
    if (pending.size) return Promise.reject(new Error("Worker busy"));
    return new Promise((resolve, reject) => {
      const id = ++serial,
        abort = () => worker.postMessage({ id, method: "abort" });
      signal?.addEventListener("abort", abort, { once: true });
      pending.set(id, {
        resolve,
        reject,
        onProgress,
        cleanup: () => signal?.removeEventListener("abort", abort),
      });
      worker.postMessage({ id, method, args });
    });
  };
  return {
    call,
    async destroy() {
      if (closed) return;
      try {
        await call("destroy");
      } finally {
        closed = true;
        worker.terminate();
        for (const p of pending.values()) {
          p.cleanup();
          p.reject(new Error("Worker terminated"));
        }
        pending.clear();
      }
    },
  };
}
