// Diagnostic ONLY: hold the PyTorch quaternion features fixed to isolate later modules.
// Never used in test:cpu or test:gpu acceptance.
import { fixture, read } from "./fixtures.mjs";
import { CPUAlgebra } from "../cpu/algebra.js";
import { loadWeights, decodeTensor, diffStats } from "../shared/weights.js";
import { denoise } from "../shared/network.js";
import { writeFile } from "node:fs/promises";
const { c, get, features } = await fixture("fp32"),
  o = new CPUAlgebra(),
  w = {};
const { config } = await loadWeights("weights/fp32", {
  read,
  onTensor: (n, t) => (w[n] = o.upload(t.shape, decodeTensor(t), true)),
});
const L = c.length;
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
const rows = {};
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
      if (n === "orientations") t.data.set(get("step2.orientations"));
      const k = "step2." + n;
      if (c.tensors[k]) {
        rows[n] = diffStats(t.data, get(k));
        console.log(n, rows[n]);
      }
    },
  },
);
await writeFile(
  new URL("../.cache/isolated-components.json", import.meta.url),
  JSON.stringify(rows, null, 2),
);
