export function adapterMetadata(adapter) {
  const i = adapter.info ?? {};
  const adapterInfo = {
    vendor: i.vendor ?? "",
    architecture: i.architecture ?? "",
    device: i.device ?? "",
    description: i.description ?? "",
  };
  const isFallbackAdapter =
    i.isFallbackAdapter === true ||
    adapter.isFallbackAdapter === true ||
    /swiftshader|llvmpipe|lavapipe|warp|software/i.test(
      Object.values(adapterInfo).join(" "),
    );
  return { adapterInfo, isFallbackAdapter };
}
export async function acquireDevice({
  device,
  allowSoftwareFallback = true,
  useF16 = true,
  signal,
} = {}) {
  signal?.throwIfAborted();
  if (device) {
    const m = adapterMetadata({ info: device.adapterInfo });
    if (
      !allowSoftwareFallback &&
      (m.isFallbackAdapter || !m.adapterInfo.vendor)
    )
      throw Error("Cannot verify external device is hardware WebGPU");
    return { device, ...m, ownsDevice: false };
  }
  if (!globalThis.navigator?.gpu) throw Error("WebGPU unavailable");
  let adapter, m;
  for (let attempt = 0; attempt < 3; attempt++) {
    adapter = null;
    for (let i = 0; i < 3 && !adapter; i++) {
      signal?.throwIfAborted();
      adapter = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance",
      });
      if (!adapter) await new Promise((r) => setTimeout(r, 100));
    }
    if (!adapter) adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw Error("No WebGPU adapter");
    m = adapterMetadata(adapter);
    if (!allowSoftwareFallback && (m.isFallbackAdapter || !m.adapterInfo.vendor))
      throw Error(
        "Hardware WebGPU required: software or unidentified adapter rejected",
      );
    signal?.throwIfAborted();
    const requiredFeatures =
      useF16 && adapter.features.has("shader-f16") ? ["shader-f16"] : [];
    const requiredLimits = {};
    for (const k of [
      "maxBufferSize",
      "maxStorageBufferBindingSize",
      "maxStorageBuffersPerShaderStage",
    ])
      requiredLimits[k] = adapter.limits[k];
    try {
      device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
      break;
    } catch (err) {
      if (
        attempt === 2 ||
        !/external Instance reference|vkCreateDevice|VK_ERROR_INITIALIZATION_FAILED/i.test(
          String(err?.message ?? err),
        )
      )
        throw err;
      await new Promise((r) => setTimeout(r, 150));
    }
  }
  if (signal?.aborted) {
    device.destroy();
    signal.throwIfAborted();
  }
  return { device, ...m, ownsDevice: true };
}
