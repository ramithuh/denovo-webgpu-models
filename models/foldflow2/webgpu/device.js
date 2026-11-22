export async function acquire({
  device,
  useF16 = true,
  allowSoftwareFallback = true,
} = {}) {
  let adapter, info;
  if (!device) {
    if (!globalThis.navigator?.gpu)
      throw Error("WebGPU requires HTTPS or localhost and a supported browser");
    for (let i = 0; i < 3 && !adapter; i++) {
      adapter = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance",
      });
      if (!adapter) await new Promise((r) => setTimeout(r, 100));
    }
    adapter ??= await navigator.gpu.requestAdapter();
    if (!adapter) throw Error("WebGPU adapter unavailable");
    info = adapter.info ?? (await adapter.requestAdapterInfo?.());
  } else info = device.adapterInfo;
  const metadata = {
    vendor: info?.vendor ?? "",
    architecture: info?.architecture ?? "",
    device: info?.device ?? "",
    description: info?.description ?? "",
    isFallbackAdapter: info?.isFallbackAdapter ?? adapter?.isFallbackAdapter,
  };
  const fallback =
    metadata.isFallbackAdapter === true ||
    /swiftshader|llvmpipe|lavapipe|software|\bwarp\b/i.test(
      Object.values(metadata).join(" "),
    );
  if (!allowSoftwareFallback && (fallback || !metadata.vendor))
    throw Error(
      "Hardware WebGPU required; software or unidentified adapter rejected",
    );
  const ownsDevice = !device;
  if (!device)
    device = await adapter.requestDevice({
      requiredFeatures:
        useF16 && adapter.features.has("shader-f16") ? ["shader-f16"] : [],
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBuffersPerShaderStage: Math.min(
          12,
          adapter.limits.maxStorageBuffersPerShaderStage,
        ),
      },
    });
  return {
    device,
    ownsDevice,
    adapterInfo: metadata,
    isFallbackAdapter: fallback,
  };
}
