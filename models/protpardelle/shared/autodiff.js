// Reverse-mode coordinate gradients for upstream reconstruction guidance.
// CPU and GPU execute the same graph; GPU derivatives are WGSL dispatches.
import { denoise } from "./denoiser.js";
import { moments, momentsWGSL } from "./moments.js";
const f = Math.fround,
  lit = (x) => Number(x).toExponential(12);
export function kernel(ops, rows, cols, inputs, code, fn) {
  if (ops.device) {
    const out = ops.tensor(rows, cols);
    ops.dispatch(
      `let i=id.x;if(i>=${rows * cols}u){return;}let r=i/${cols}u;let c=i%${cols}u;${code}`,
      inputs.map((t) => t.buffer).concat(out.buffer),
      [Math.ceil((rows * cols) / 64)],
    );
    return out;
  }
  const arrays = inputs.map((t) => t.data),
    data = new Float32Array(rows * cols);
  for (let i = 0; i < data.length; i++)
    data[i] = fn(i, Math.floor(i / cols), i % cols, arrays);
  return { data, rows, cols };
}
function linearBack(ops, x, g, prefix) {
  const w = ops.weight(prefix + ".weight"),
    [o, k] = w.shape;
  if (!ops.device) {
    const out = new Float32Array(x.rows * k);
    for (let r = 0; r < x.rows; r++)
      for (let c = 0; c < k; c++) {
        let sum = 0;
        for (let j = 0; j < o; j++)
          sum = f(sum + g.data[r * o + j] * w.data[j * k + c]);
        out[r * k + c] = sum;
      }
    return { data: out, rows: x.rows, cols: k };
  }
  const out = ops.tensor(x.rows, k),
    type = w.dtype === "F32" ? "f32" : "u32",
    load =
      w.dtype === "F32"
        ? "return W[i];"
        : w.dtype === "F16"
          ? "return unpack2x16float(W[i/2u])[i%2u];"
          : `let v=i32(((W[i/4u]>>((i%4u)*8u))&255u)<<24u)>>24;return f32(v)*S[i/${k}u];`;
  const code = `@group(0) @binding(0)var<storage,read> G:array<f32>;@group(0) @binding(1)var<storage,read> W:array<${type}>;@group(0) @binding(2)var<storage,read> S:array<f32>;@group(0) @binding(3)var<storage,read_write> Y:array<f32>;fn weight(i:u32)->f32{${load}}@compute @workgroup_size(64)fn main(@builtin(global_invocation_id)id:vec3<u32>){let i=id.x;if(i>=${x.rows * k}u){return;}let r=i/${k}u;let c=i%${k}u;var sum=0.0;for(var j=0u;j<${o}u;j++){sum=fma(G[r*${o}u+j],weight(j*${k}u+c),sum);}Y[i]=sum+0.0*S[0];}`;
  ops.dispatch(
    code,
    [g.buffer, w.buffer, w.scale ?? ops.zero, out.buffer],
    [Math.ceil((x.rows * k) / 64)],
  );
  return out;
}
function normBack(ops, x, g, prefix) {
  const w = ops.weights[prefix + ".gamma"] ?? ops.weight(prefix + ".weight"),
    d = x.cols;
  if (!ops.device) {
    const out = new Float32Array(x.data.length),
      scale = f(1 / d);
    for (let r = 0; r < x.rows; r++) {
      const { mean, variance } = moments(x.data, r * d, d),
        inv = f(1 / f(Math.sqrt(f(variance + f(1e-5))))),
        ds = new Float32Array(4),
        db = new Float32Array(4);
      for (let j = 0; j < d; j++) {
        const v = g.data[r * d + j];
        ds[j % 4] = f(ds[j % 4] + f(f(v * x.data[r * d + j]) * w.data[j]));
        db[j % 4] = f(db[j % 4] + f(v * w.data[j]));
      }
      const a = Array.from(ds).reduce((a, b) => f(a + b)),
        b = Array.from(db).reduce((a, b) => f(a + b));
      const coefficient = f(
          f(f(f(f(b * mean) - a) * inv) * inv) * f(inv * scale),
        ),
        constant = f(f(-coefficient * mean) - f(f(b * inv) * scale));
      for (let c = 0; c < d; c++) {
        const i = r * d + c;
        out[i] = f(
          f(f(f(inv * g.data[i]) * w.data[c]) + f(coefficient * x.data[i])) +
            constant,
        );
      }
    }
    return { data: out, rows: x.rows, cols: d };
  }
  const out = ops.tensor(x.rows, d);
  ops.dispatch(
    `let r=id.x;if(r>=${x.rows}u){return;} ${momentsWGSL(d)}
    let inv=1.0/sqrt(variance+0.00001);var ds=vec4<f32>(0.0);var db=vec4<f32>(0.0);
    for(var j=0u;j<${d}u;j++){
      let dy=b1[r*${d}u+j];let lane=j%4u;
      ds[lane]+=(dy*b0[r*${d}u+j])*b2[j];db[lane]+=dy*b2[j];
    }
    let a=((ds.x+ds.y)+ds.z)+ds.w;let b=((db.x+db.y)+db.z)+db.w;
    let coefficient=(((b*mean-a)*inv)*inv)*(inv/f32(${d}));
    let constant=-coefficient*mean-(b*inv)/f32(${d});
    for(var c=0u;c<${d}u;c++){
      let i=r*${d}u+c;b3[i]=((inv*b1[i])*b2[c]+coefficient*b0[i])+constant;
    }`,
    [x.buffer, g.buffer, w.buffer, out.buffer],
    [Math.ceil(x.rows / 64)],
  );
  return out;
}
function attentionBack(ops, out, g) {
  const {
      queries: q,
      keys: k,
      probs: p,
      kv,
      input,
      config,
      prefix,
    } = out.attentionCache,
    n = out.rows,
    d = out.cols,
    h = config.model.struct_model.uvit.n_heads,
    dh = d / h;
  const dot = kernel(
    ops,
    h * n,
    n,
    [g, kv],
    `let head=r/${n}u;let row=r%${n}u;var sum=0.0;for(var a=0u;a<${dh}u;a++){sum+=b0[row*${d}u+head*${dh}u+a]*b1[c*${2 * d}u+${d}u+head*${dh}u+a];}b2[i]=sum;`,
    (i, r, c, [g, v]) => {
      const head = Math.floor(r / n),
        row = r % n;
      let sum = 0;
      for (let a = 0; a < dh; a++)
        sum += g[row * d + head * dh + a] * v[c * 2 * d + d + head * dh + a];
      return sum;
    },
  );
  const ds = kernel(
    ops,
    h * n,
    n,
    [dot, p],
    `var mean=0.0;for(var j=0u;j<${n}u;j++){mean+=b0[r*${n}u+j]*b1[r*${n}u+j];}b2[i]=b1[i]*(b0[i]-mean);`,
    (i, r, c, [dot, p]) => {
      let sum = 0;
      for (let j = 0; j < n; j++) sum += dot[r * n + j] * p[r * n + j];
      return p[i] * (dot[i] - sum);
    },
  );
  const gq = kernel(
    ops,
    n,
    d,
    [ds, k],
    `let head=c/${dh}u;let a=c%${dh}u;var sum=0.0;for(var j=0u;j<${n}u;j++){sum+=b0[(head*${n}u+r)*${n}u+j]*b1[j*${d}u+head*${dh}u+a];}b2[i]=sum;`,
    (i, r, c, [ds, k]) => {
      const head = Math.floor(c / dh);
      let s = 0;
      for (let j = 0; j < n; j++)
        s += ds[(head * n + r) * n + j] * k[j * d + c];
      return s;
    },
  );
  const gk = kernel(
    ops,
    n,
    d,
    [ds, q],
    `let head=c/${dh}u;let a=c%${dh}u;var sum=0.0;for(var j=0u;j<${n}u;j++){sum+=b0[(head*${n}u+j)*${n}u+r]*b1[j*${d}u+head*${dh}u+a];}b2[i]=sum;`,
    (i, r, c, [ds, q]) => {
      const head = Math.floor(c / dh);
      let s = 0;
      for (let j = 0; j < n; j++)
        s += ds[(head * n + j) * n + r] * q[j * d + c];
      return s;
    },
  );
  const gv = kernel(
    ops,
    n,
    d,
    [p, g],
    `let head=c/${dh}u;var sum=0.0;for(var j=0u;j<${n}u;j++){sum+=b0[(head*${n}u+j)*${n}u+r]*b1[j*${d}u+c];}b2[i]=sum;`,
    (i, r, c, [p, g]) => {
      const head = Math.floor(c / dh);
      let s = 0;
      for (let j = 0; j < n; j++) s += p[(head * n + j) * n + r] * g[j * d + c];
      return s;
    },
  );
  let rawq = gq,
    rawk = gk;
  const freqs = ops.weights[prefix + "rope.freqs"];
  if (freqs) {
    const pos = ops.fromArray(input.residueIndex, n, 1),
      chain = ops.fromArray(input.chainIndex ?? new Float32Array(n), n, 1);
    const undo = (x) =>
      kernel(
        ops,
        n,
        d,
        [x, pos, chain, freqs],
        `let angle=b1[r]*b3[(c%${dh}u)/2u]+b2[r]*b3[0];let cs=preciseSinCos(angle);b4[i]=b0[i]*cs.y+select(b0[r*${d}u+(c^1u)],-b0[r*${d}u+(c^1u)],c%2u==1u)*cs.x;`,
        (i, r, c, [x, pos, chain, freq]) => {
          const angle = f(
            f(pos[r] * freq[c % dh >> 1]) + f(chain[r] * freq[0]),
          );
          return (
            x[i] * Math.cos(angle) +
            (c % 2 ? -1 : 1) * x[r * d + (c ^ 1)] * Math.sin(angle)
          );
        },
      );
    rawq = undo(gq);
    rawk = undo(gk);
  }
  rawq = kernel(
    ops,
    n,
    d,
    [rawq],
    `b1[i]=b0[i]*${lit(dh ** -0.5)};`,
    (i, r, c, [x]) => x[i] * f(dh ** -0.5),
  );
  const rawkv = kernel(
    ops,
    n,
    2 * d,
    [rawk, gv],
    `if(c<${d}u){b2[i]=b0[r*${d}u+c];}else{b2[i]=b1[r*${d}u+c-${d}u];}`,
    (i, r, c, [k, v]) => (c < d ? k[r * d + c] : v[r * d + c - d]),
  );
  return [rawq, rawkv];
}
export function denoiseWithGradient(
  ops,
  config,
  input,
  target,
  lossMask,
  capture = () => {},
) {
  const tracked = new Set(),
    tape = [];
  const methods = new Set([
    "features",
    "linear",
    "norm",
    "silu",
    "add",
    "mask",
    "modulate",
    "attention",
    "output",
  ]);
  const proxy = new Proxy(ops, {
    get(object, name) {
      if (name === "beginBlock" || name === "endBlock") return () => {};
      const original = object[name];
      if (typeof original !== "function") return original;
      if (!methods.has(name)) return original.bind(object);
      return (...args) => {
        const out = original.apply(object, args);
        if (
          name === "features" ||
          name === "output" ||
          args.some((v) => tracked.has(v))
        ) {
          tracked.add(out);
          tape.push({ name, args, out });
        }
        return out;
      };
    },
  });
  const result = denoise(proxy, config, input, capture),
    n = input.seqMask.length,
    a = config.model.struct_model.n_atoms,
    sd = config.data.sigma_data;
  const targetTensor = ops.fromArray(target, n, a * 3),
    maskTensor = ops.fromArray(lossMask, n, a);
  const seed = kernel(
    ops,
    n,
    a * 3,
    [result, targetTensor, maskTensor],
    `b3[i]=2.0*(b0[i]-b1[i])*b2[r*${a}u+c/3u];`,
    (i, r, c, [x, t, m]) => 2 * (x[i] - t[i]) * m[r * a + Math.floor(c / 3)],
  );
  const grads = new Map([[result, seed]]);
  let coordinateGrad;
  const add = (x, g) => {
    if (!tracked.has(x) || !g) return;
    grads.set(x, grads.has(x) ? ops.add(grads.get(x), g) : g);
  };
  for (const record of tape.reverse()) {
    const { name, args, out } = record,
      g = grads.get(out);
    if (!g) continue;
    const [x, y, z] = args;
    if (input.debugGradients && name === "linear") capture("grad." + y, g);
    if (name === "linear") add(x, linearBack(ops, x, g, y));
    else if (name === "norm") add(x, normBack(ops, x, g, y));
    else if (name === "silu")
      add(
        x,
        kernel(
          ops,
          x.rows,
          x.cols,
          [x, g],
          `let s=1.0/(1.0+exp(-b0[i]));b2[i]=b1[i]*s*(1.0+b0[i]*(1.0-s));`,
          (i, r, c, [x, g]) => {
            const s = 1 / (1 + Math.exp(-x[i]));
            return g[i] * s * (1 + x[i] * (1 - s));
          },
        ),
      );
    else if (name === "add") {
      add(x, g);
      add(y, g);
    } else if (name === "mask") add(x, ops.mask(g, y));
    else if (name === "modulate") {
      const offset = z === "gate" ? 2 * x.cols : 0;
      add(
        x,
        kernel(
          ops,
          x.rows,
          x.cols,
          [g, y],
          `b2[i]=b0[i]*(1.0+b1[(r%${y.rows}u)*${y.cols}u+c+${offset}u]);`,
          (i, r, c, [g, t]) =>
            g[i] * (1 + t[(r % y.rows) * y.cols + c + offset]),
        ),
      );
    } else if (name === "attention") {
      const [q, kv] = attentionBack(ops, out, g);
      add(x, q);
      add(y, kv);
    } else if (name === "output") {
      const levels = ops.fromArray(input.noiseLevel, n, 1),
        mask = ops.fromArray(input.seqMask, n, 1);
      add(
        x,
        kernel(
          ops,
          n,
          a * 3,
          [g, levels, mask],
          `let s=max(0.000001,b1[r]);b3[i]=b0[r*${a * 3}u+(c%${a}u)*3u+c/${a}u]*(s*${lit(sd)}/sqrt(s*s+${lit(sd * sd)}))*b2[r];`,
          (i, r, c, [g, s, m]) =>
            g[r * a * 3 + (c % a) * 3 + Math.floor(c / a)] *
            ((s[r] * sd) / Math.sqrt(s[r] * s[r] + sd * sd)) *
            m[r],
        ),
      );
      coordinateGrad = kernel(
        ops,
        n,
        a * 3,
        [g, levels, mask],
        `let s=max(0.000001,b1[r]);b3[i]=b0[i]*(${lit(sd * sd)}/(s*s+${lit(sd * sd)}))*b2[r];`,
        (i, r, c, [g, s, m]) =>
          ((g[i] * sd * sd) / (s[r] * s[r] + sd * sd)) * m[r],
      );
    } else if (name === "features") {
      const levels = ops.fromArray(input.noiseLevel, n, 1),
        width = out.cols;
      const featureGrad = kernel(
        ops,
        n,
        a * 3,
        [g, levels],
        `let s=max(0.000001,b1[r]);b2[i]=b0[r*${width}u+(c%3u)*${a}u+c/3u]/sqrt(s*s+${lit(sd * sd)});`,
        (i, r, c, [g, s]) =>
          g[r * width + (c % 3) * a + Math.floor(c / 3)] /
          Math.sqrt(s[r] * s[r] + sd * sd),
      );
      coordinateGrad = ops.add(coordinateGrad, featureGrad);
    }
  }
  return { coords: result, gradient: coordinateGrad };
}
