import { sizeOf } from "./loader.js";
export { sizeOf };
export function strides(shape) {
  let s = 1;
  return shape
    .map((_, i) => i)
    .reverse()
    .map((i) => {
      const v = s;
      s *= shape[i];
      return v;
    })
    .reverse();
}
export function tensor(storage, shape, stride = strides(shape), offset = 0) {
  return { storage, shape, stride, offset };
}
export function contiguous(t) {
  let s = 1;
  for (let i = t.shape.length - 1; i >= 0; i--) {
    if (t.shape[i] > 1 && t.stride[i] !== s) return false;
    s *= t.shape[i];
  }
  return true;
}
export function physical(t, i) {
  let p = t.offset;
  for (let d = t.shape.length - 1; d >= 0; d--) {
    p += (i % t.shape[d]) * t.stride[d];
    i = Math.floor(i / t.shape[d]);
  }
  return p;
}
export const axis = (d, n) => (d < 0 ? d + n : d);
export function view(t, shape) {
  if (sizeOf(shape) !== sizeOf(t.shape)) throw Error("Invalid reshape");
  if (contiguous(t)) return tensor(t.storage, shape, strides(shape), t.offset);
  // PyTorch computeStride: preserve each contiguous subspace, including expanded axes.
  const ns = new Array(shape.length);
  let vd = shape.length - 1,
    chunk = t.stride.at(-1),
    tn = 1,
    vn = 1;
  for (let td = t.shape.length - 1; td >= 0; td--) {
    tn *= t.shape[td];
    if (
      td === 0 ||
      (t.shape[td - 1] !== 1 && t.stride[td - 1] !== tn * chunk)
    ) {
      while (vd >= 0 && (vn < tn || shape[vd] === 1)) {
        ns[vd] = vn * chunk;
        vn *= shape[vd];
        vd--;
      }
      if (vn !== tn) return null;
      if (td > 0) {
        chunk = t.stride[td - 1];
        tn = 1;
        vn = 1;
      }
    }
  }
  return vd === -1 ? tensor(t.storage, shape, ns, t.offset) : null;
}
export function indexExpression(t, index = "i", gpu = false) {
  let den = 1;
  const terms = [];
  for (let d = t.shape.length - 1; d >= 0; d--) {
    if (t.shape[d] > 1 && t.stride[d]) {
      const q =
        den === 1
          ? index
          : gpu
            ? `(${index}/${den}u)`
            : `Math.floor(${index}/${den})`;
      const mod = d === 0 ? q : `(${q}%${t.shape[d]}${gpu ? "u" : ""})`;
      terms.push(
        t.stride[d] === 1 ? mod : `(${mod}*${t.stride[d]}${gpu ? "u" : ""})`,
      );
    }
    den *= t.shape[d];
  }
  if (t.offset) terms.push(String(t.offset) + (gpu ? "u" : ""));
  return terms.join("+") || (gpu ? "0u" : "0");
}
export function broadcast(t, shape) {
  const stride = shape.map((n, i) => {
    const j = i - (shape.length - t.shape.length);
    if (j < 0) return 0;
    if (t.shape[j] === 1) return 0;
    if (t.shape[j] !== n)
      throw Error(`Broadcast mismatch ${t.shape} -> ${shape}`);
    return t.stride[j];
  });
  return tensor(t.storage, shape, stride, t.offset);
}
