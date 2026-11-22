const fetchOriginal = globalThis.fetch,
  cachesOriginal = globalThis.caches,
  getDirectory = navigator.storage.getDirectory.bind(navigator.storage);
let offline = false;
globalThis.fetch = (...a) => {
  if (offline) throw Error("Network disabled by storage test");
  return fetchOriginal(...a);
};
self.addEventListener("message", (event) => {
  const { cmd, id, args } = event.data;
  if (cmd !== "test-storage") return;
  event.stopImmediatePropagation();
  offline = args.offline ?? false;
  Object.defineProperty(globalThis, "caches", {
    configurable: true,
    get() {
      if (args.fail === "cache")
        throw new DOMException("Private Mode", "SecurityError");
      return cachesOriginal;
    },
  });
  navigator.storage.getDirectory = async () => {
    if (args.fail === "opfs")
      throw new DOMException("Private Mode", "SecurityError");
    return getDirectory();
  };
  self.postMessage({ id, result: true });
});
await import("../webgpu/worker.js");

self.postMessage({ ready: true });
