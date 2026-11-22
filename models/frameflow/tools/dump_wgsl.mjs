import { browserTask } from "../test/browser.mjs";
import { mkdir, writeFile } from "node:fs/promises";
import { root } from "../test/helpers.mjs";
const shaders = await browserTask((page) =>
  page.evaluate(async () => {
    const source = {},
      original = GPUDevice.prototype.createShaderModule;
    GPUDevice.prototype.createShaderModule = function (d) {
      source[d.label] = d.code;
      return original.call(this, d);
    };
    const { createFrameFlowEngine } = await import("/webgpu/index.js"),
      { fixture } = await import("/test/parity.js");
    const read = async (u) => (await fetch(u)).arrayBuffer();
    for (const precision of ["fp32", "fp16", "int8"]) {
      const e = await createFrameFlowEngine({
        baseUrl: "/weights",
        precision,
        allowSoftwareFallback: false,
      });
      const f = await fixture(precision + "_L32_T2", read, "/");
      await e.sample({ initial: f.input, steps: 2 });
      e.destroy();
    }
    const e = await createFrameFlowEngine({
      baseUrl: "/weights",
      precision: "fp16",
      allowSoftwareFallback: false,
      useF16: true,
    });
    await e.forward((await fixture("fp16_L32_sc0", read, "/")).input);
    e.destroy();
    GPUDevice.prototype.createShaderModule = original;
    return source;
  }),
);
await mkdir(root + ".cache/wgsl", { recursive: true });
for (const [label, code] of Object.entries(shaders))
  await writeFile(root + ".cache/wgsl/" + label + ".wgsl", code);
console.log(Object.keys(shaders));
