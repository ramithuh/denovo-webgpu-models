export function createGenie2Worker({
  url = new URL("./worker.js", import.meta.url),
} = {}) {
  const worker = new Worker(url, { type: "module" }),
    pending = new Map();
  let seq = 0,
    closed = false;
  worker.onmessage = ({ data }) => {
    const p = pending.get(data.id);
    if (!p) return;
    if (data.event === "step") {
      p.onStep?.(data.step);
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
      p.reject(new Error(e.message));
    }
    pending.clear();
  };
  const call = (method, options = {}, handlers = {}) =>
    new Promise((resolve, reject) => {
      if (closed) {
        reject(new Error("Worker closed"));
        return;
      }
      const id = ++seq;
      const abort = () => worker.postMessage({ method: "abort" });
      if (handlers.signal?.aborted) {
        reject(handlers.signal.reason);
        return;
      }
      handlers.signal?.addEventListener("abort", abort, { once: true });
      pending.set(id, {
        resolve,
        reject,
        onStep: handlers.onStep,
        cleanup: () => handlers.signal?.removeEventListener("abort", abort),
      });
      worker.postMessage({ id, method, options });
    });
  return {
    init: (options) => call("init", options),
    forward: (options, handlers) => call("forward", options, handlers),
    sample: (options, handlers) => call("sample", options, handlers),
    memoryUsage: () => call("memoryUsage"),
    async destroy() {
      if (closed) return;
      try {
        await call("destroy");
      } finally {
        closed = true;
        worker.terminate();
        for (const p of pending.values()) {
          p.cleanup();
          p.reject(new Error("Worker closed"));
        }
        pending.clear();
      }
    },
  };
}
