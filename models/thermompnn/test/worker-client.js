export async function makeWorker() {
  const worker = new Worker(new URL("../webgpu/worker.js", import.meta.url), {
    type: "module",
  });
  let sequence = 0;
  const pending = new Map();
  let readyResolve, readyReject;
  const ready = new Promise((resolve, reject) => {
    readyResolve = resolve;
    readyReject = reject;
  });
  worker.onmessage = ({ data }) => {
    if (data.ready) {
      readyResolve();
      return;
    }
    const p = pending.get(data.id);
    if (!p) return;
    if (data.progress) {
      p.progress?.(data.progress);
      return;
    }
    pending.delete(data.id);
    if (data.error)
      p.reject(
        Object.assign(new Error(data.error), {
          name: data.name,
          memory: data.memory,
        }),
      );
    else p.resolve(data.result);
  };
  worker.onerror = (e) => {
    readyReject(e);
    for (const p of pending.values()) p.reject(e);
  };
  await ready;
  return {
    worker,
    call(command, { options, text, onProgress } = {}) {
      const id = ++sequence;
      return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject, progress: onProgress });
        worker.postMessage({ id, command, options, text });
      });
    },
    cancel() {
      worker.postMessage({ command: "cancel" });
    },
    close() {
      worker.terminate();
    },
  };
}
