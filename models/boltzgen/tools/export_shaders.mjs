// Validate actual modular model shader variants without storing an inference program.
import { readFile, writeFile } from "node:fs/promises";
import { tensor, sizeOf } from "../shared/tensor.js";
import { shader } from "../webgpu/shaders.js";
import { BoltzGen } from "../shared/model.js";
import { createFeatures, derivedFeatures } from "../shared/input.js";
import { randomInputs } from "../shared/random.js";
const codes = new Set();
class ShaderBackend {
  constructor() {
    this.live = new Set();
  }
  upload(data, shape, persistent = false) {
    const s = { persistent, refs: 0 };
    this.live.add(s);
    return tensor(s, shape);
  }
  retain(t) {
    t.storage.refs++;
  }
  release(t) {
    if (!t.storage.persistent) this.live.delete(t.storage);
  }
  compute(kind, inputs, shapes, attrs = {}, inplace = false) {
    const outputs = shapes.map((s, i) =>
      inplace && i === 0 ? attrs.target : this.upload(null, s),
    );
    if (kind === "cat" || kind === "stack") {
      const out = outputs[0],
        d = attrs.dim;
      let offset = 0;
      for (const input of inputs) {
        const stride =
          kind === "stack" ? out.stride.filter((_, i) => i !== d) : out.stride;
        this.compute(
          "copy",
          [input],
          [input.shape],
          {
            target: tensor(
              out.storage,
              input.shape,
              stride,
              offset * out.stride[d],
            ),
          },
          true,
        );
        offset += kind === "stack" ? 1 : input.shape[d];
      }
    } else codes.add(shader(kind, inputs, outputs, attrs).code);
    return outputs.length === 1 ? outputs[0] : outputs;
  }
}
for (const precision of ["fp32", "fp16", "int8"]) {
  const manifest = JSON.parse(
    await readFile(
      new URL(`../weights/diverse/${precision}/manifest.json`, import.meta.url),
      "utf8",
    ),
  );
  for (const L of [8, 20, 24, 64]) {
    const backend = new ShaderBackend(),
      weights = new Map();
    for (const [name, t] of Object.entries(manifest.tensors)) {
      if (name.endsWith(".__scale")) continue;
      const v = backend.upload(null, t.shape, true);
      if (t.dtype === "float16") v.storage.encoding = "F16";
      if (t.dtype === "int8") {
        v.storage.encoding = "I8";
        v.storage.columns = t.shape[1];
        v.storage.scale = backend.upload(null, [t.shape[0]], true).storage;
      }
      weights.set(name, v);
    }
    const features = createFeatures({ L }),
      f = {};
    for (const [name, t] of features)
      f[name.slice(2)] = backend.upload(t.data, t.shape);
    for (const [name, t] of Object.entries(derivedFeatures(features)))
      f[name] = backend.upload(t.data, t.shape);
    const random = [...randomInputs(f.atom_pad_mask.shape[1], 3).values()].map(
      (t) => backend.upload(t.data, t.shape),
    );
    await new BoltzGen(backend, weights).forward(f, random, { numSteps: 3 });
    console.log(precision, L, codes.size);
  }
}
await writeFile(
  new URL("../.cache/shaders.json", import.meta.url),
  JSON.stringify([...codes]),
);
