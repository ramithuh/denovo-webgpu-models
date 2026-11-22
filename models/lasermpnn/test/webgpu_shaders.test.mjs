import test from "node:test";
import assert from "node:assert/strict";
import { browserRun, gpuSetup } from "./browser.mjs";
test("Every f32 and shader-f16 kernel compiles with zero Dawn validation errors", async () => {
  const result = await browserRun(
    `const {GpuBackend}=await import('/webgpu/lasermpnn/backend.js');const {OP_SHADERS}=await import('/webgpu/lasermpnn/shaders.js');${gpuSetup}device.pushErrorScope('validation');const coreAdapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});const coreDevice=await coreAdapter.requestDevice();coreDevice.pushErrorScope('validation');const a=await GpuBackend.create(coreDevice,{},{});a.destroy();const coreError=await coreDevice.popErrorScope();coreDevice.destroy();if(coreError)throw Error(coreError.message);if(!device.features.has('shader-f16'))throw Error('shader-f16 not available on test adapter');const b=await GpuBackend.create(device,{}, {shaderF16:true});
  for (const width of [1, 7, 128, 259]) {
    const rows = 3;
    const values = Float32Array.from({ length: rows * width }, (_, i) =>
      i < width ? 4 : Math.sin(i * 0.3) + (i % 9));
    const scale = Float32Array.from({ length: width }, (_, i) => 0.5 + i % 3);
    const bias = Float32Array.from({ length: width }, (_, i) => i % 5 - 2);
    const key = "norm" + width;
    b.weights[key + ".weight"] = { data: scale, shape: [width] };
    b.weights[key + ".bias"] = { data: bias, shape: [width] };
    const result = await b.read(b.norm(b.from(values, rows, width), key));
    for (let row = 0; row < rows; row++) {
      const src = values.subarray(row * width, (row + 1) * width);
      const mean = src.reduce((s, v) => s + v, 0) / width;
      const variance = src.reduce((s, v) => s + (v - mean) ** 2, 0) / width;
      for (let col = 0; col < width; col++) {
        const expected = (src[col] - mean) / Math.sqrt(variance + 1e-5) * scale[col] + bias[col];
        if (!(Math.abs(result[row * width + col] - expected) < 2e-5)) {
          throw Error("LayerNorm mismatch at width " + width + ", row " + row + ", column " + col);
        }
      }
    }
  }
 b.destroy();const error=await device.popErrorScope();if(error)throw Error(error.message);return {shaderVariants:Object.keys(OP_SHADERS).length+2};`,
  );
  assert.ok(result.ok, result.error);
  console.log(result);
});
