// Diagnostic only. Oracle quaternion injection isolates later neural modules.
// Not imported by any acceptance suite; not an end-to-end parity result.
import { loadFixture, log } from "./browser-fixture.js";
import { GPUAlgebra } from "../webgpu/algebra.js";
import { acquireDevice } from "../webgpu/device.js";
import { loadWeights, diffStats } from "../shared/weights.js";
import { denoise } from "../shared/network.js";
export async function run() {
  const { c, get, features } = await loadFixture("fp32"),
    { device, adapterInfo } = await acquireDevice({
      allowSoftwareFallback: false,
    }),
    o = new GPUAlgebra(device),
    w = {},
    rows = {};
  try {
    const { config } = await loadWeights("/weights/fp32", {
        onTensor: (n, t) => (w[n] = o.uploadWeight(t)),
      }),
      L = c.length;
    const ft = Object.fromEntries(
      Object.entries(features).map(([k, v]) => [
        k,
        o.tensor(
          k === "aatype"
            ? [L, 20]
            : k === "atom_positions"
              ? [L, 3]
              : k === "fixed_structure_mask"
                ? [L, L]
                : [L],
          v,
        ),
      ]),
    );
    await denoise(
      o,
      w,
      config,
      {
        rots: o.tensor([L, 3, 3], get("step2.input_rots")),
        trans: o.tensor([L, 3], get("initial")),
      },
      ft,
      2,
      2,
      {
        onStage: async (n, t) => {
          if (n === "orientations") {
            await o.read(t);
            device.queue.writeBuffer(t.buffer, 0, get("step2.orientations"));
          }
          const k = "step2." + n;
          if (c.tensors[k]) {
            rows[n] = diffStats(await o.read(t), get(k));
            await log({ n, ...rows[n] });
          }
        },
      },
    );
    return {
      diagnosticOnly: true,
      quaternionFeaturesFromOracle: true,
      adapterInfo,
      rows,
    };
  } finally {
    o.destroy();
    device.destroy();
  }
}
