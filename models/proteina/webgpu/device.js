export function rejectSoftware(info, fallback = false) {
  if (
    fallback ||
    info?.isFallbackAdapter ||
    /swiftshader|llvmpipe|lavapipe|software|warp|microsoft basic/i.test(
      [info?.vendor, info?.architecture, info?.device, info?.description].join(
        " ",
      ),
    )
  )
    throw Error("Hardware WebGPU required; software/fallback adapter rejected");
}
export async function acquire({
  device,
  allowSoftwareFallback = true,
  useF16 = false,
  signal,
} = {}) {
  signal?.throwIfAborted();
  let adapter,
    info = device?.adapterInfo;
  if (!device) {
    if (!globalThis.navigator?.gpu) throw Error("WebGPU unavailable");
    for (let attempt = 0; attempt < 3 && !adapter; attempt++) {
      adapter = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance",
      });
      signal?.throwIfAborted();
    }
    if (!adapter) adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw Error("No WebGPU adapter");
    info = adapter.info ?? (await adapter.requestAdapterInfo?.()) ?? {};
  } else if (!allowSoftwareFallback && !info)
    throw Error("Cannot verify externally supplied device adapterInfo");
  const isFallbackAdapter = !!(
    adapter?.isFallbackAdapter ||
    info?.isFallbackAdapter ||
    /swiftshader|llvmpipe|lavapipe|software|warp/i.test(
      [info?.vendor, info?.architecture, info?.description].join(" "),
    )
  );
  if (!allowSoftwareFallback) rejectSoftware(info, isFallbackAdapter);
  const owned = !device;
  if (owned) {
    const limits = {};
    for (const key of ["maxBufferSize", "maxStorageBufferBindingSize"])
      limits[key] = adapter.limits[key];
    device = await adapter.requestDevice({
      requiredLimits: limits,
      requiredFeatures:
        useF16 && adapter.features.has("shader-f16") ? ["shader-f16"] : [],
    });
  }
  try {
    signal?.throwIfAborted();
    if (useF16 && !device.features.has("shader-f16"))
      throw Error("useF16 requires shader-f16");
  } catch (e) {
    if (owned) device.destroy();
    throw e;
  }
  return {
    device,
    owned,
    adapterInfo: {
      vendor: info?.vendor ?? "",
      architecture: info?.architecture ?? "",
      device: info?.device ?? "",
      description: info?.description ?? "",
      isFallbackAdapter,
    },
    isFallbackAdapter,
  };
}
