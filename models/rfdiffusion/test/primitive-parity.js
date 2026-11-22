import { metric } from "./parity.js";
import {
  normLast,
  sphericalHarmonics,
  cross,
  quaternionRotation,
} from "../shared/geometry.js";
export async function primitiveParity(o, data) {
  const shape = (x) => (Array.isArray(x) ? [x.length, ...shape(x[0])] : []),
    tensor = (x) => o.tensor(shape(x), x.flat(Infinity)),
    out = {};
  for (const [key, d] of Object.entries(data)) {
    const x = tensor(d.x);
    let y;
    if (key.startsWith("norm")) y = normLast(o, x);
    else if (key.startsWith("softmax"))
      y = o.softmax(x, key.includes("Axis") ? 1 : -1);
    else if (key === "scalarScale") y = o.mul(x, d.scale);
    else if (key === "fma")
      y = o.kernel(
        x.shape,
        [x, tensor(d.b), tensor(d.c)],
        "o[i]=rfFma(a0[i],a1[i],a2[i]);",
        "o[i]=fma(a0[i],a1[i],a2[i]);",
      );
    else if (key === "dglSoftmax") y = o.softmax(x, 0, true);
    else if (key === "quaternion") y = quaternionRotation(o, x);
    else if (key === "sh") y = sphericalHarmonics(o, x);
    else if (key === "cross") y = cross(o, x, tensor(d.b));
    else if (key === "matmul3") y = o.einsum("nij,njk->nik", x, tensor(d.b));
    else if (key === "group") {
      const ys = [];
      for (let g = 0; g < 2; g++)
        ys.push(
          o.groupNorm(
            o.slice(x, 1, g * 32, (g + 1) * 32),
            tensor(d.w.slice(g * 32, (g + 1) * 32)),
            tensor(d.b.slice(g * 32, (g + 1) * 32)),
          ),
        );
      y = o.cat(ys, 1);
    } else throw Error("Unknown primitive");
    const m = metric(await o.read(y), d.y.flat(Infinity));
    if (
      !(m.relRMS < 1e-6) ||
      (["fma", "scalarScale"].includes(key) && m.maxAbs !== 0)
    )
      throw Error(`Primitive ${key}: ${m.relRMS}`);
    out[key] = m;
    o.clear();
  }
  return out;
}
