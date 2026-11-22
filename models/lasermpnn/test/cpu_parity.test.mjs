import test from "node:test";
import { CpuBackend } from "../cpu/lasermpnn/backend.js";
import { LaserModel } from "../shared/lasermpnn/model.js";
import { makeGeometry } from "../shared/lasermpnn/geometry.js";
import { localWeights, oracle, batchFromOracle, compare } from "./helpers.mjs";
test("CPU FP32 full upstream holo encoder, coupled AA/chi decoding and hydrogen coordinates", async () => {
  const weights = await localWeights(),
    o = await oracle(),
    batch = batchFromOracle(o),
    backend = new CpuBackend(weights.tensors),
    model = new LaserModel(backend, weights.manifest.config);
  const result = await model.run(batch, {
    trace: true,
    uniforms: o.read("rng.categorical_uniforms"),
  });
  for (const [name, value] of Object.entries(result.traces))
    compare(value, o.read(name), name);
  compare(
    result.sequence,
    o.read("output.sampled_sequence_indices"),
    "sequence",
  );
  compare(result.chiAngles, o.read("output.sampled_chi_degrees"), "chiAngles");
  const built = makeGeometry(weights.tensors).build(
    batch.backbone,
    result.sequence,
    result.chiAngles,
  );
  compare(
    built.coords,
    o.read("output.coords"),
    "sidechains including hydrogens",
  );
  makeGeometry(weights.tensors).cleanup(built, result.sequence);
  compare(
    built.coords,
    o.read("output.cleaned_coords"),
    "CLI hydrogen postprocessing",
  );
});
