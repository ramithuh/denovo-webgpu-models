import { createGenie2Engine } from "../webgpu/index.js";
import { diffStats } from "../shared/weights.js";
import { loadFixture, log } from "./browser-fixture.js";
export async function run() {
  const { c, get, features } = await loadFixture("fp32"),
    engine = await createGenie2Engine({
      baseUrl: "/weights/fp32",
      allowSoftwareFallback: false,
    });
  await log(engine.adapterInfo);
  const rows = {};
  const start = performance.now();
  try {
    const r = await engine.forward({
      coordinates: get("initial"),
      rotations: get("step2.input_rots"),
      features,
      timestep: 2,
      steps: 2,
      onStage: async (n, t) => {
        const k = "step2." + n;
        if (c.tensors[k]) {
          const m = diffStats(t, get(k));
          rows[n] = m;
          await log({ stage: n, ...m });
        }
      },
    });
    rows.elapsed = performance.now() - start;
    return {
      adapterInfo: engine.adapterInfo,
      rows,
      memory: engine.memoryUsage(),
    };
  } finally {
    engine.destroy();
  }
}
