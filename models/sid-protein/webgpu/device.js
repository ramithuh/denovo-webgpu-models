export function adapterMetadata(adapter, info) {
  const data = {
    vendor: info?.vendor ?? "",
    architecture: info?.architecture ?? "",
    device: info?.device ?? "",
    description: info?.description ?? "",
  };
  const isFallbackAdapter = !!(
    adapter?.isFallbackAdapter ||
    info?.isFallbackAdapter ||
    /swiftshader|llvmpipe|lavapipe|warp|software/i.test(JSON.stringify(data))
  );
  return { adapterInfo: data, isFallbackAdapter };
}
export async function acquireDevice({
  device,
  useF16 = true,
  allowSoftwareFallback = true,
} = {}) {
  if (device) {
    const metadata = adapterMetadata(null, device.adapterInfo);
    if (
      !allowSoftwareFallback &&
      (!device.adapterInfo || metadata.isFallbackAdapter)
    )
      throw new Error(
        "Strict tests require verifiable hardware adapter metadata",
      );
    return { device, ownsDevice: false, ...metadata };
  }
  if (!globalThis.navigator?.gpu)
    throw new Error(
      "WebGPU requires HTTPS or localhost and a compatible browser",
    );
  let lastErr;
  for (let attempt = 0; attempt < 4; attempt++) {
    let adapter;
    for (let i = 0; i < 2 && !adapter; i++) {
      adapter = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance",
      });
      if (!adapter) await new Promise((r) => setTimeout(r, 50));
    }
    if (!adapter) adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No WebGPU adapter available");
    const info = adapter.info ?? (await adapter.requestAdapterInfo?.()),
      metadata = adapterMetadata(adapter, info);
    if (!allowSoftwareFallback && metadata.isFallbackAdapter)
      throw new Error(
        "Hardware WebGPU required; software/fallback adapter rejected",
      );
    try {
      const d = await adapter.requestDevice({
        requiredFeatures:
          useF16 && adapter.features.has("shader-f16") ? ["shader-f16"] : [],
        requiredLimits: {
          maxBufferSize: adapter.limits.maxBufferSize,
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        },
      });
      return { device: d, ownsDevice: true, ...metadata };
    } catch (e) {
      lastErr = e;
      await new Promise((r) => setTimeout(r, 100));
    }
  }
  throw lastErr;
}
