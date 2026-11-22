import { normalization } from "./moments.js";
// Small tensor algebra shared by the independent JS and WGSL backends.
export const sizeOf = (s) => s.reduce((a, b) => a * b, 1);
export const stridesOf = (s) => s.map((_, i) => sizeOf(s.slice(i + 1)));
export class Algebra {
  tensor(shape, data) {
    return this.upload(shape, Float32Array.from(data));
  }
  reshape(x, shape) {
    if (shape.includes(-1))
      shape = shape.map((n) =>
        n === -1 ? x.size / sizeOf(shape.filter((n) => n !== -1)) : n,
      );
    if (sizeOf(shape) !== x.size) throw Error("reshape");
    return { ...x, shape };
  }
  constant(shape, value = 0) {
    return this.tensor(shape, new Float32Array(sizeOf(shape)).fill(value));
  }
  expr(shape, inputs, expression) {
    const rank = shape.length,
      strides = stridesOf(shape),
      vars = shape.map((d, i) => `c${i}`);
    const coords = shape.map((d, i) => [
      `const c${i}=Math.floor(i/${strides[i]})%${d};`,
      `let c${i}=(i/${strides[i]}u)%${d}u;`,
    ]);
    const accesses = inputs.map((t, n) => {
      const ts = stridesOf(t.shape);
      return `a${n}[${t.shape.map((d, j) => (d === 1 ? "0" : `${vars[rank - t.shape.length + j]}*${ts[j]}`)).join("+") || "0"}]`;
    });
    let e = expression;
    accesses.forEach((a, j) => (e = e.replaceAll(`$${j}`, a)));
    const js = e
      .replaceAll("exp(", "rfExp(")
      .replaceAll("sqrt(", "Math.sqrt(")
      .replaceAll("max(", "Math.max(")
      .replaceAll("abs(", "Math.abs(")
      .replaceAll("sin(", "Math.sin(")
      .replaceAll("cos(", "Math.cos(");
    return this.kernel(
      shape,
      inputs,
      coords.map((x) => x[0]).join("") + `o[i]=${js};`,
      coords.map((x) => x[1]).join("") +
        `o[i]=${e.replaceAll("exp(", "rfExp(")};`,
    );
  }
  element(expression, ...xs) {
    const ts = xs.filter((x) => typeof x !== "number"),
      rank = Math.max(...ts.map((x) => x.shape.length));
    const shape = Array(rank).fill(1);
    for (const t of ts)
      t.shape.forEach((d, j) => {
        const k = rank - t.shape.length + j;
        if (shape[k] !== 1 && d !== 1 && shape[k] !== d)
          throw Error("broadcast");
        shape[k] = Math.max(shape[k], d);
      });
    let n = 0;
    const e = expression.replace(/\$(\d+)/g, (_, k) =>
      typeof xs[k] === "number"
        ? `${Math.fround(xs[k]).toExponential()}`
        : `$${n++}`,
    );
    return this.expr(shape, ts, e);
  }
  add(a, b) {
    return this.element("$0+$1", a, b);
  }
  sub(a, b) {
    return this.element("$0-$1", a, b);
  }
  mul(a, b) {
    return this.element("$0*$1", a, b);
  }
  div(a, b) {
    return this.element("rfdiv($0,$1)", a, b);
  }
  relu(a) {
    return this.element("max($0,0.0)", a);
  }
  sigmoid(a) {
    return this.kernel(
      a.shape,
      [a],
      "o[i]=1/Math.fround(1+rfExp(-a0[i]));",
      "o[i]=rfdiv(1.0,1.0+rfExp(-a0[i]));",
    );
  }
  unary(a, e) {
    return this.expr(a.shape, [a], e);
  }
  einsum(eq, ...xs) {
    const [left, right] = eq.replaceAll(" ", "").split("->"),
      terms = left.split(","),
      dims = {};
    terms.forEach((s, j) => {
      if (s.length !== xs[j].shape.length) throw Error(`einsum rank ${eq}`);
      [...s].forEach((c, k) => {
        const d = xs[j].shape[k];
        if (dims[c] && dims[c] !== d) throw Error(`einsum dimension ${eq}`);
        dims[c] = d;
      });
    });
    const reduce = Object.keys(dims)
        .filter((c) => !right.includes(c))
        .sort(),
      shape = [...right].map((c) => dims[c]),
      os = stridesOf(shape),
      rs = stridesOf(reduce.map((c) => dims[c]));
    const coord = (letters, strides, index, wgsl) =>
      [...letters]
        .map((c, k) =>
          wgsl
            ? `let e_${c}=(${index}/${strides[k]}u)%${dims[c]}u;`
            : `const e_${c}=Math.floor(${index}/${strides[k]})%${dims[c]};`,
        )
        .join("");
    const expr = terms
      .map((s, j) => {
        const st = stridesOf(xs[j].shape);
        return `a${j}[${[...s].map((c, k) => `e_${c}*${st[k]}`).join("+") || "0"}]`;
      })
      .join("*");
    const n = sizeOf(reduce.map((c) => dims[c]));
    return this.kernel(
      shape,
      xs,
      coord(right, os, "i", false) +
        `let v=0;for(let r=0;r<${n};r++){${coord(reduce.join(""), rs, "r", false)}v=${terms.length === 2 ? "rfFma(" + expr.replace("*a1[", ",a1[") + ",v)" : "Math.fround(v+" + expr + ")"};}o[i]=v;`,
      coord(right, os, "i", true) +
        `var v=0.0;for(var r=0u;r<${n}u;r++){${coord(reduce.join(""), rs, "r", true)}${
          terms.length === 2
            ? "v=fma(" +
              terms
                .map((term, j) => {
                  const st = stridesOf(xs[j].shape);
                  return `a${j}[${[...term].map((c, k) => `e_${c}*${st[k]}`).join("+") || "0"}]`;
                })
                .join(",") +
              ",v);"
            : "v=v+" + expr + ";"
        }}o[i]=v;`,
    );
  }
  linear(x, w, b) {
    const shape = x.shape.slice(0, -1),
      K = x.shape.at(-1),
      M = w.shape[0],
      inputs = b ? [x, w, b] : [x, w];
    return this.kernel(
      [...shape, M],
      inputs,
      `const row=Math.floor(i/${M}),col=i%${M};let v=${b ? "a2[col]" : "0"};for(let k=0;k<${K};k++)v=rfFma(a0[row*${K}+k],a1[col*${K}+k],v);o[i]=v;`,
      `let row=i/${M}u;let col=i%${M}u;var v=${b ? "a2[col]" : "0.0"};for(var k=0u;k<${K}u;k++){v=fma(a0[row*${K}u+k],a1[col*${K}u+k],v);}o[i]=v;`,
    );
  }
  permute(x, order) {
    const labels = "abcdefghijk".slice(0, x.shape.length);
    return this.einsum(`${labels}->${order.map((k) => labels[k]).join("")}`, x);
  }
  slice(x, axis, start, end) {
    const shape = x.shape.slice();
    shape[axis] = end - start;
    const os = stridesOf(shape),
      ts = stridesOf(x.shape);
    const idx = shape
      .map(
        (d, k) => `((i/${os[k]}u)%${d}u+${k === axis ? start : 0}u)*${ts[k]}u`,
      )
      .join("+");
    return this.kernel(
      shape,
      [x],
      `o[i]=a0[${idx.replaceAll(/(\d+)u/g, "$1").replaceAll(/\(i\/(\d+)\)/g, "Math.floor(i/$1)")}];`,
      `o[i]=a0[${idx}];`,
    );
  }
  cat(xs, axis = -1) {
    if (axis < 0) axis += xs[0].shape.length;
    const shape = xs[0].shape.slice();
    shape[axis] = xs.reduce((n, x) => n + x.shape[axis], 0);
    const inner = sizeOf(shape.slice(axis + 1)),
      outer = sizeOf(shape.slice(0, axis)),
      width = shape[axis];
    let start = 0,
      js = "",
      wg = "";
    xs.forEach((x, j) => {
      const n = x.shape[axis];
      const index = `outer*${n * inner}+((c-${start})*${inner})+inner`;
      js += `${j ? "else " : ""}if(c<${start + n})o[i]=a${j}[${index}];`;
      wg += `${j ? "else " : ""}if(c<${start + n}u){o[i]=a${j}[${index}];}`;
      start += n;
    });
    return this.kernel(
      shape,
      xs,
      `const outer=Math.floor(i/${width * inner}),c=Math.floor(i/${inner})%${width},inner=i%${inner};` +
        js,
      `let outer=i/${width * inner}u;let c=(i/${inner}u)%${width}u;let inner=i%${inner}u;` +
        wg,
    );
  }
  norm(x, w, b, eps = 1e-5) {
    return normalization(this, x, w, b, false, eps);
  }
  groupNorm(x, w, b, eps = 1e-5) {
    return normalization(this, x, w, b, true, eps);
  }
  softmax(x, axis = -1, sequential = false) {
    if (axis < 0) axis += x.shape.length;
    const n = x.shape[axis],
      inner = sizeOf(x.shape.slice(axis + 1)),
      total = n * inner;
    const dgl = sequential;
    sequential = sequential || inner > 1;
    const js = `const base=Math.floor(i/${total})*${total}+i%${inner},f=Math.fround;let m=-Infinity;for(let j=0;j<${n};j++)m=Math.max(m,a0[base+j*${inner}]);const lanes=new Float32Array(4);for(let j=0;j<${n};j++){const v=${dgl ? "rfScalarExp(f(a0[base+j*" + inner + "]-m))" : "rfExp(f(a0[base+j*" + inner + "]-m))"},k=${sequential ? "0" : "j%4"};lanes[k]=f(lanes[k]+v);}const s=${sequential ? "lanes[0]" : n < 4 ? "f(f(lanes[0]+lanes[1])+lanes[2])" : "f(f(lanes[0]+lanes[2])+f(lanes[1]+lanes[3]))"};o[i]=${dgl ? "rfScalarExp(f(a0[i]-m))/s" : sequential ? "rfExp(f(a0[i]-m))/s" : "rfExp(f(a0[i]-m))*f(1/s)"};`;
    const wg = `let base=(i/${total}u)*${total}u+i%${inner}u;var m=-3.402823e38;for(var j=0u;j<${n}u;j++){m=max(m,a0[base+j*${inner}u]);}var lanes:array<f32,4>;for(var j=0u;j<${n}u;j++){let v=rfExp(a0[base+j*${inner}u]-m);let k=${sequential ? "0u" : "j%4u"};lanes[k]=lanes[k]+v;}let s=${sequential ? "lanes[0]" : n < 4 ? "(lanes[0]+lanes[1])+lanes[2]" : "(lanes[0]+lanes[2])+(lanes[1]+lanes[3])"};o[i]=${sequential ? "rfdiv(rfExp(a0[i]-m),s)" : "rfExp(a0[i]-m)*rfdiv(1.0,s)"};`;
    return this.kernel(x.shape, [x], js, wg);
  }
  gather(x, index) {
    const width = sizeOf(x.shape.slice(1));
    return this.kernel(
      [...index.shape, ...x.shape.slice(1)],
      [x, index],
      `o[i]=a0[Math.round(a1[Math.floor(i/${width})])*${width}+i%${width}];`,
      `o[i]=a0[u32(a1[i/${width}u])*${width}u+i%${width}u];`,
    );
  }
}
