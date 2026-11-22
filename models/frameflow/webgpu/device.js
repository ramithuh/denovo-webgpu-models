export function adapterMetadata(info = {}) {
  const data = {
    vendor: info.vendor ?? "",
    architecture: info.architecture ?? "",
    device: info.device ?? "",
    description: info.description ?? "",
    isFallbackAdapter: !!info.isFallbackAdapter,
  };
  data.isFallbackAdapter ||=
    /swiftshader|llvmpipe|lavapipe|warp|software|basic render/i.test(
      Object.values(data).join(" "),
    );
  return data;
}
export async function acquireDevice({
  device,
  allowSoftwareFallback = true,
  useF16 = false,
  signal,
} = {}) {
  signal?.throwIfAborted();
  if (device) {
    const info = adapterMetadata(device.adapterInfo);
    if (!allowSoftwareFallback && (!info.vendor || info.isFallbackAdapter))
      throw Error(
        "Hardware WebGPU adapter required; external device lacks hardware metadata",
      );
    return {
      device,
      owned: false,
      adapterInfo: info,
      isFallbackAdapter: info.isFallbackAdapter,
    };
  }
  if (!globalThis.navigator?.gpu) throw Error("WebGPU unavailable");
  let adapter;
  for (let i = 0; i < 3 && !adapter; i++) {
    signal?.throwIfAborted();
    adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) adapter = await navigator.gpu.requestAdapter();
  }
  if (!adapter) throw Error("Hardware WebGPU adapter required: No adapter");
  const info = adapterMetadata(adapter.info);
  if (!allowSoftwareFallback && info.isFallbackAdapter)
    throw Error("Hardware WebGPU required; software adapter rejected");
  const requiredFeatures =
    useF16 && adapter.features.has("shader-f16") ? ["shader-f16"] : [];
  device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBuffersPerShaderStage: 8,
    },
  });
  return {
    device,
    owned: true,
    adapterInfo: info,
    isFallbackAdapter: info.isFallbackAdapter,
  };
}
