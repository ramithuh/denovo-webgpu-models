import { tensor, view, broadcast, sizeOf, axis } from "./tensor.js";

// Eager tensor algebra shared by the JavaScript and WGSL implementations.
// Shapes come from live inputs; no operation tape or captured program is used.
export class Ops {
  constructor(backend, weights) {
    this.backend = backend;
    this.weights = weights;
    this.scopes = [];
  }
  track(t) {
    for (const v of Array.isArray(t) ? t : [t])
      if (v?.storage && !v.storage.persistent)
        this.scopes.at(-1)?.add(v.storage);
    return t;
  }
  calc(kind, inputs, shape, attrs = {}) {
    return this.track(this.backend.compute(kind, inputs, [shape], attrs));
  }
  data(data, shape) {
    return this.track(
      this.backend.upload(Float32Array.from(data), shape, false),
    );
  }
  fill(shape, value = 0) {
    return this.calc("fill", [], shape, { value });
  }
  weight(name, optional = false) {
    const w =
      this.weights.get(name) ??
      this.weights.get(
        name.replace(".token_transformer.", ".token_transformer_layers.0."),
      ) ??
      this.weights.get("w:" + name);
    if (!w && !optional) throw Error(`Missing weight ${name}`);
    return w;
  }
  tidy(fn) {
    const allocated = new Set();
    this.scopes.push(allocated);
    let result;
    try {
      result = fn();
      return result;
    } finally {
      this.scopes.pop();
      const keep = storages(result);
      for (const s of allocated)
        if (keep.has(s)) this.scopes.at(-1)?.add(s);
        else this.backend.release({ storage: s });
    }
  }
  reshape(t, shape) {
    shape = [...shape];
    const j = shape.indexOf(-1);
    if (j >= 0) shape[j] = sizeOf(t.shape) / -sizeOf(shape);
    return (
      view(t, shape) ?? tensor(this.calc("copy", [t], t.shape).storage, shape)
    );
  }
  permute(t, dims) {
    return tensor(
      t.storage,
      dims.map((d) => t.shape[d]),
      dims.map((d) => t.stride[d]),
      t.offset,
    );
  }
  transpose(t, a = -2, b = -1) {
    const dims = t.shape.map((_, i) => i);
    a = axis(a, dims.length);
    b = axis(b, dims.length);
    [dims[a], dims[b]] = [dims[b], dims[a]];
    return this.permute(t, dims);
  }
  unsqueeze(t, d) {
    d = axis(d, t.shape.length + 1);
    const sh = [...t.shape],
      st = [...t.stride];
    sh.splice(d, 0, 1);
    st.splice(d, 0, 0);
    return tensor(t.storage, sh, st, t.offset);
  }
  expand(t, shape) {
    return broadcast(t, shape);
  }
  slice(t, d, start, end) {
    d = axis(d, t.shape.length);
    const sh = [...t.shape];
    sh[d] = end - start;
    return tensor(t.storage, sh, t.stride, t.offset + start * t.stride[d]);
  }
  select(t, d, i) {
    d = axis(d, t.shape.length);
    return tensor(
      t.storage,
      t.shape.filter((_, j) => j !== d),
      t.stride.filter((_, j) => j !== d),
      t.offset + i * t.stride[d],
    );
  }
  point(kind, ...args) {
    const shape = args
      .filter((t) => t?.shape)
      .reduce((s, t) => broadcastShape(s, t.shape), []);
    return this.calc("pointwise", args, shape, { kind, alpha: 1 });
  }
  add(a, b) {
    return this.point("add", a, b);
  }
  sub(a, b) {
    return this.point("sub", a, b);
  }
  mul(a, b) {
    return this.point("mul", a, b);
  }
  div(a, b) {
    return this.point("div", a, b);
  }
  sigmoid(a) {
    return this.point("sigmoid", a);
  }
  silu(a) {
    return this.point("silu", a);
  }
  relu(a) {
    return this.point("relu", a);
  }
  sum(t, dims, keep = false, kind = "sum") {
    dims = (Array.isArray(dims) ? dims : [dims]).map((d) =>
      axis(d, t.shape.length),
    );
    const sh = t.shape.flatMap((s, i) =>
      dims.includes(i) ? (keep ? [1] : []) : [s],
    );
    return this.calc("reduce", [t], sh, { dims, kind });
  }
  norm(t, p, eps = 1e-5) {
    const w = t.shape.at(-1),
      sh = [...t.shape.slice(0, -1), 1];
    return this.track(
      this.backend.compute(
        "norm",
        [
          t,
          p ? this.weight(p + ".weight", true) : null,
          p ? this.weight(p + ".bias", true) : null,
        ],
        [t.shape, sh, sh],
        { width: w, epsilon: eps },
      ),
    )[0];
  }
  mm(a, b, bias) {
    const batch = broadcastShape(a.shape.slice(0, -2), b.shape.slice(0, -2));
    if (a.shape.at(-1) !== b.shape.at(-2))
      throw Error(`Matmul ${a.shape} @ ${b.shape}`);
    a = broadcast(a, [...batch, ...a.shape.slice(-2)]);
    b = broadcast(b, [...batch, ...b.shape.slice(-2)]);
    return this.calc(
      "matmul",
      [a, b, ...(bias ? [bias] : [])],
      [...batch, a.shape.at(-2), b.shape.at(-1)],
    );
  }
  linear(t, p) {
    const w = this.weight(p + ".weight"),
      bias = this.weight(p + ".bias", true);
    const x = this.reshape(t, [-1, t.shape.at(-1)]),
      y = this.mm(x, this.transpose(w), bias);
    return this.reshape(y, [...t.shape.slice(0, -1), w.shape[0]]);
  }
  seq(t, p) {
    return this.linear(this.norm(t, p + ".0"), p + ".1");
  }
  embed(t, p) {
    const w = this.weight(p + ".weight");
    return this.calc("embedding", [w, t], [...t.shape, w.shape[1]]);
  }
  cat(ts, d = -1) {
    d = axis(d, ts[0].shape.length);
    const shape = [...ts[0].shape];
    shape[d] = ts.reduce((n, t) => n + t.shape[d], 0);
    return this.calc("cat", ts, shape, { dim: d });
  }
  softmax(t, d = -1) {
    return this.calc("softmax", [t], t.shape, { dim: axis(d, t.shape.length) });
  }
  cdist(a, b = a) {
    return this.calc(
      "cdist",
      [a, b],
      [...a.shape.slice(0, -2), a.shape.at(-2), b.shape.at(-2)],
    );
  }
  onehot(t, n) {
    return this.calc("onehot", [t], [...t.shape, n], { width: n });
  }
  keys(t) {
    const n = sizeOf(t.shape.slice(0, -1)),
      d = t.shape.at(-1);
    return this.calc("window", [this.reshape(t, [1, n, d])], [n / 32, 128, d], {
      queries: 32,
      keys: 128,
    });
  }
  collect(...roots) {
    const keep = storages(roots),
      live = this.backend.live ?? this.backend.storages;
    for (const s of [...live])
      if (!s.persistent && !keep.has(s)) this.backend.release({ storage: s });
  }
}
export function storages(x, out = new Set()) {
  if (x?.storage) out.add(x.storage);
  else if (x instanceof Map) for (const t of x.values()) storages(t, out);
  else if (Array.isArray(x)) x.forEach((t) => storages(t, out));
  else if (x && typeof x === "object" && !ArrayBuffer.isView(x))
    Object.values(x).forEach((t) => storages(t, out));
  return out;
}
export function broadcastShape(a, b) {
  const n = Math.max(a.length, b.length);
  return Array.from({ length: n }, (_, i) => {
    const x = a[i - n + a.length] ?? 1,
      y = b[i - n + b.length] ?? 1;
    if (x !== y && x !== 1 && y !== 1) throw Error(`Broadcast ${a} vs ${b}`);
    return Math.max(x, y);
  });
}
