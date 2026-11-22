export async function requestHardwareAdapter() {
  if (!globalThis.navigator?.gpu)
    throw new Error("Hardware WebGPU unavailable");
  // Chromium on this driver can return null during the first native Vulkan probe.
  // Retry initialization briefly, then fail. Never request a software fallback.
  for (let attempt = 0; attempt < 3; attempt++) {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (adapter) {
      const info = adapter.info;
      const fallback = info.isFallbackAdapter ?? adapter.isFallbackAdapter;
      if (
        fallback === true ||
        /swiftshader|llvmpipe|software/i.test(
          [info.vendor, info.architecture, info.device, info.description].join(
            " ",
          ),
        )
      )
        throw new Error("Software WebGPU adapters are unsupported");
      if (fallback === false && info.vendor) return adapter;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(
    "Hardware WebGPU adapter required (three native probes failed)",
  );
}
