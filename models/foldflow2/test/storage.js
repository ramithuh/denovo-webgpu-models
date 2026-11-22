import { loadBundle } from "../shared/loader.js";
export async function storageTests() {
  const baseUrl = new URL("/.cache/fixtures/loader", location.href).href,
    result = [];
  for (const cache of ["cache", "opfs"]) {
    const a = await loadBundle({ baseUrl, cache });
    if (a.weights.probe.data[7] !== 7) throw Error("Storage cold mismatch");
    const b = await loadBundle({
      baseUrl,
      cache,
      read: () => {
        throw Error("Warm cache missed");
      },
    });
    if (b.weights.probe.data[5] !== 5) throw Error("Storage warm mismatch");
    result.push(cache + " cold/warm");
  }
  const cacheOpen = caches.open.bind(caches),
    getDir = navigator.storage.getDirectory.bind(navigator.storage);
  try {
    caches.open = async () => {
      throw new DOMException("Private browsing", "SecurityError");
    };
    navigator.storage.getDirectory = async () => {
      throw new DOMException("Private browsing", "SecurityError");
    };
    for (const cache of ["cache", "opfs"]) {
      let reads = 0;
      await loadBundle({
        baseUrl,
        cache,
        read: async (u) => {
          reads++;
          return (await fetch(u)).arrayBuffer();
        },
      });
      if (reads !== 2) throw Error("Storage denied fallback was not exercised");
      result.push(cache + " storage-denied fallback");
    }
  } finally {
    caches.open = cacheOpen;
    navigator.storage.getDirectory = getDir;
  }
  return result;
}
