import test from "node:test";
import { readFile, writeFile } from "node:fs/promises";
import { CPUAlgebra } from "../cpu/algebra.js";
import { primitiveParity } from "./primitive-parity.js";
import { browserTask } from "./browser.mjs";
import { root } from "./helpers.mjs";
test("PyTorch float32 primitives on CPU and hardware WGSL", async () => {
  const data = JSON.parse(
    await readFile(root + ".cache/fixtures/primitives.json"),
  );
  const cpu = await primitiveParity(new CPUAlgebra(), data);
  const gpu = await browserTask((page) =>
    page.evaluate(async () => {
      const { requestHardwareAdapter } = await import("/webgpu/device.js");
      const { GPUAlgebra } = await import("/webgpu/algebra.js");
      const { primitiveParity } = await import("/test/primitive-parity.js");
      const adapter = await requestHardwareAdapter(),
        device = await adapter.requestDevice(),
        o = new GPUAlgebra(device);
      try {
        const r = await fetch("/.cache/fixtures/primitives.json");
        if (!r.ok) throw Error("Missing primitive oracle");
        return await primitiveParity(o, await r.json());
      } finally {
        o.destroy();
        device.destroy();
      }
    }),
  );
  await writeFile(
    root + "fixtures/numerics_metrics.json",
    JSON.stringify({ cpu, gpu }, null, 2) + "\n",
  );
});
