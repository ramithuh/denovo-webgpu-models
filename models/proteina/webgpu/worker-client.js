export function createProteinaWorker({
  url = new URL("./worker.js", import.meta.url),
} = {}) {
  const worker = new Worker(url, { type: "module" }),
    pending = new Map();
  let seq = 0,
    closed = false;
  worker.onmessage = ({ data }) => {
    const p = pending.get(data.id);
    if (!p) return;
    if (data.progress) {
      try {
        p.onProgress?.(data.progress);
      } catch (e) {
        worker.postMessage({ cmd: "cancel", args: { id: data.id } });
        p.reject(e);
      }
      return;
    }
    pending.delete(data.id);
    p.cleanup();
    if (data.error)
      p.reject(Object.assign(new Error(data.error), { name: data.name }));
    else p.resolve(data.result);
  };
  worker.onerror = (e) => {
    for (const p of pending.values()) {
      p.cleanup();
      p.reject(Error(e.message));
    }
    pending.clear();
  };
  function call(cmd, args = {}, options = {}) {
    if (closed) return Promise.reject(Error("Worker closed"));
    const { signal, onProgress } = options;
    signal?.throwIfAborted();
    return new Promise((resolve, reject) => {
      const id = ++seq,
        abort = () => worker.postMessage({ cmd: "cancel", args: { id } });
      signal?.addEventListener("abort", abort, { once: true });
      pending.set(id, {
        resolve,
        reject,
        onProgress,
        cleanup: () => signal?.removeEventListener("abort", abort),
      });
      worker.postMessage({ id, cmd, args });
    });
  }
  return {
    init: (args, options) => call("init", args, options),
    forward: (coordinates, args = {}, options = {}) =>
      call("forward", { ...args, coordinates }, options),
    sample: (args, options) => call("sample", args, options),
    memoryUsage: () => call("memory"),
    async destroy() {
      if (closed) return;
      try {
        await call("destroy");
      } finally {
        closed = true;
        worker.terminate();
        for (const p of pending.values()) {
          p.cleanup();
          p.reject(Error("Worker closed"));
        }
        pending.clear();
      }
    },
  };
}
