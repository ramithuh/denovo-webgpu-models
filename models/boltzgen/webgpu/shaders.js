import {
  broadcast,
  indexExpression,
  sizeOf,
  strides,
  tensor,
} from "../shared/tensor.js";
const fl = (x) => {
  if (x === Infinity) return "3.4028234663852886e38";
  if (x === -Infinity) return "-3.4028234663852886e38";
  if (!Number.isFinite(x)) throw Error(`Invalid shader literal ${x}`);
  const s = String(x);
  return /[.eE]/.test(s) ? s : s + ".0";
};
const ui = (x) => `${x}u`;

export function shader(kind, inputs, outputs, attrs = {}) {
  const bindings = [],
    map = new Map();
  function bind(t, write = false) {
    if (!t || typeof t === "number") return;
    let b = map.get(t.storage);
    if (!b) {
      b = { storage: t.storage, name: "b" + bindings.length, write };
      bindings.push(b);
      map.set(t.storage, b);
    }
    b.write ||= write;
    return b.name;
  }
  inputs.forEach((t) => bind(t));
  outputs.forEach((t) => bind(t, true));
  const get = (t, i = "i") => {
    if (typeof t === "number") return fl(t);
    const name = bind(t),
      index = indexExpression(t, i, true);
    if (t.storage.encoding === "F16")
      return `unpack2x16float(${name}[(${index})/2u])[(${index})%2u]`;
    if (t.storage.encoding === "I8") {
      const scale = bind({ storage: t.storage.scale });
      return `(f32(i32((${name}[(${index})/4u]>>(((${index})%4u)*8u))<<24u)>>24u)*${scale}[(${index})/${ui(t.storage.columns)}])`;
    }
    return `${name}[${index}]`;
  };
  const put = (t, i, v) =>
    `${bind(t, true)}[${indexExpression(t, i, true)}]=${v};`;
  const [a, b, c] = inputs,
    out = outputs[0],
    shape = out.shape,
    count = sizeOf(shape);
  let body = "",
    extra = "",
    workgroup = "64",
    dispatch = [Math.ceil(count / 64), 1, 1],
    invocations = count;
  const loop = (name, n, code) =>
    `for(var ${name}=0u;${name}<${ui(n)};${name}++){${code}}`;
  switch (kind) {
    case "onehot":
      body = put(
        out,
        "i",
        `select(0.0,1.0,u32(${get(a, `(i/${ui(attrs.width)})`)})==i%${ui(attrs.width)})`,
      );
      break;
    case "window": {
      const d = shape[2],
        n = a.shape[1];
      body =
        `let win=i/${ui(128 * d)};let key=(i/${ui(d)})%128u;let j=i32(win*32u+key)-48;var value=0.0;if(j>=0&&j<${n}){value=${get(a, `(u32(j)*${ui(d)}+i%${ui(d)})`)};}` +
        put(out, "i", "value");
      break;
    }
    case "fill":
      body = put(out, "i", fl(attrs.value));
      break;
    case "range":
      body = put(out, "i", `${fl(attrs.start)}+f32(i)*${fl(attrs.step)}`);
      break;
    case "eye":
      body = put(
        out,
        "i",
        `select(0.0,1.0,i/${ui(attrs.width)}==i%${ui(attrs.width)})`,
      );
      break;
    case "copy":
    case "cast": {
      let value = get(broadcast(a, shape));
      if (kind === "cast") {
        if (attrs.dtype === "torch.bool")
          value = `select(0.0,1.0,${value}!=0.0)`;
        else if (/torch.int/.test(attrs.dtype)) value = `trunc(${value})`;
      }
      body = put(out, "i", value);
      break;
    }
    case "pointwise": {
      const vals = inputs.map((t) =>
          t === null
            ? null
            : get(typeof t === "number" ? t : broadcast(t, shape)),
        ),
        [x, y, z] = vals;
      let v;
      switch (attrs.kind) {
        case "add":
          v = `${x}+${y}*${fl(attrs.alpha)}`;
          break;
        case "sub":
          v = `${x}-${y}*${fl(attrs.alpha)}`;
          break;
        case "rsub":
          v = `${y}-${x}*${fl(attrs.alpha)}`;
          break;
        case "mul":
          v = `${x}*${y}`;
          break;
        case "div":
          v = `${x}/${y}`;
          break;
        case "pow":
          v = inputs[1] === 2 ? `${x}*${x}` : `pow(${x},${y})`;
          break;
        case "sigmoid":
          v = `1.0/(1.0+exp(-${x}))`;
          break;
        case "silu":
          v = `${x}/(1.0+exp(-${x}))`;
          break;
        case "relu":
          v = `max(0.0,${x})`;
          break;
        case "sqrt":
        case "exp":
        case "log":
        case "cos":
        case "abs":
          v = `${attrs.kind}(${x})`;
          break;
        case "reciprocal":
          v = `1.0/${x}`;
          break;
        case "neg":
          v = `-${x}`;
          break;
        case "eq":
        case "ne":
        case "lt":
        case "gt":
        case "le":
          v = `select(0.0,1.0,${x}${{ eq: "==", ne: "!=", lt: "<", gt: ">", le: "<=" }[attrs.kind]}${y})`;
          break;
        case "bitwise_and":
          v = `f32(i32(${x})&i32(${y}))`;
          break;
        case "bitwise_not":
          v = `select(0.0,1.0,${x}==0.0)`;
          break;
        case "where":
          v = `select(${z},${y},${x}!=0.0)`;
          break;
        case "clamp":
          v = `min(${z ?? fl(Infinity)},max(${y ?? fl(-Infinity)},${x}))`;
          break;
        default:
          throw Error(`Unsupported pointwise ${attrs.kind}`);
      }
      body = put(out, "i", v);
      break;
    }
    case "matmul": {
      const M = a.shape.at(-2),
        K = a.shape.at(-1),
        N = b.shape.at(-1),
        batch = count / (M * N),
        bias = c ? broadcast(c, shape) : null;
      extra =
        "var<workgroup> tileA:array<f32,256>;var<workgroup> tileB:array<f32,256>;";
      workgroup = "16,16";
      dispatch = [Math.ceil(N / 16), Math.ceil(M / 16), batch];
      body = `let m=wg.y*16u+lid.y;let n=wg.x*16u+lid.x;let h=wg.z;var acc=0.0;var correction=0.0;
        for(var block=0u;block<${ui(K)};block+=16u){
          var av=0.0;var bv=0.0;
          if(m<${ui(M)}&&block+lid.x<${ui(K)}){av=${get(a, `((h*${ui(M)}+m)*${ui(K)}+block+lid.x)`)};}
          if(n<${ui(N)}&&block+lid.y<${ui(K)}){bv=${get(b, `((h*${ui(K)}+block+lid.y)*${ui(N)}+n)`)};}
          tileA[lid.y*16u+lid.x]=av;tileB[lid.y*16u+lid.x]=bv;workgroupBarrier();
          for(var k=0u;k<16u;k++){
            let av=tileA[lid.y*16u+k];let bv=tileB[k*16u+lid.x];let product=av*bv;
            let productError=fma(av,bv,-product);let term=product-correction;let next=acc+term;
            correction=(next-acc)-term-productError;acc=next;
          }workgroupBarrier();
        }
        if(m<${ui(M)}&&n<${ui(N)}){let i=(h*${ui(M)}+m)*${ui(N)}+n;${put(out, "i", `(acc-correction)*${fl(attrs.alpha ?? 1)}${bias ? "+" + get(bias) + "*" + fl(attrs.beta ?? 1) : ""}`)}}`;
      break;
    }
    case "norm": {
      const w = attrs.width;
      invocations = count / w;
      body = `var mean=0.0;var mc=0.0;${loop("j", w, `let y=${get(a, `(i*${ui(w)}+j)`)}-mc;let t=mean+y;mc=(t-mean)-y;mean=t;`)}mean=(mean-mc)/${fl(w)};
        var variance=0.0;var vc=0.0;${loop("j", w, `let delta=${get(a, `(i*${ui(w)}+j)`)}-mean;let y=delta*delta-vc;let t=variance+y;vc=(t-variance)-y;variance=t;`)}variance-=vc;
        let inv=inverseSqrt(variance/${fl(w)}+${fl(attrs.epsilon)});
        ${put(outputs[1], "i", "mean")}${put(outputs[2], "i", "inv")}
        ${loop("j", w, put(out, `(i*${ui(w)}+j)`, `(${get(a, `(i*${ui(w)}+j)`)}-mean)*inv*${b ? get(b, "j") : "1.0"}+${c ? get(c, "j") : "0.0"}`))}`;
      break;
    }
    case "softmax": {
      const d = attrs.dim,
        w = shape[d],
        inner = sizeOf(shape.slice(d + 1));
      invocations = count / w;
      body = `let base=(i/${ui(inner)})*${ui(w * inner)}+i%${ui(inner)};var mx=${fl(-Infinity)};
        ${loop("j", w, `mx=max(mx,${get(a, `(base+j*${ui(inner)})`)});`)}
        var sum=0.0;var sc=0.0;${loop("j", w, `let y=exp(${get(a, `(base+j*${ui(inner)})`)}-mx)-sc;let t=sum+y;sc=(t-sum)-y;sum=t;`)}sum-=sc;
        ${loop("j", w, put(out, `(base+j*${ui(inner)})`, `select(exp(${get(a, `(base+j*${ui(inner)})`)}-mx)/sum,0.0,mx==${fl(-Infinity)})`))}`;
      break;
    }
    case "reduce": {
      const dims = new Set(attrs.dims),
        rv = tensor(
          a.storage,
          a.shape.filter((_, i) => dims.has(i)),
          a.stride.filter((_, i) => dims.has(i)),
          0,
        ),
        nv = tensor(
          a.storage,
          a.shape.filter((_, i) => !dims.has(i)),
          a.stride.filter((_, i) => !dims.has(i)),
          a.offset,
        ),
        rn = sizeOf(rv.shape);
      const value = `${bind(a)}[${indexExpression(nv, "i", true)}+${indexExpression(rv, "j", true)}]`;
      const init =
        attrs.kind === "min"
          ? fl(Infinity)
          : attrs.kind === "max"
            ? fl(-Infinity)
            : "0.0";
      const compensated =
        attrs.kind === "sum" || attrs.kind === "linalg_vector_norm";
      const update =
        attrs.kind === "min"
          ? `acc=min(acc,${value});`
          : attrs.kind === "max"
            ? `acc=max(acc,${value});`
            : attrs.kind === "any"
              ? `acc=select(acc,1.0,${value}!=0.0);`
              : `let y=${attrs.kind === "linalg_vector_norm" ? `${value}*${value}` : value}-correction;let t=acc+y;correction=(t-acc)-y;acc=t;`;
      body = `var acc=${init};var correction=0.0;${loop("j", rn, update)}${compensated ? "acc-=correction;" : ""}${put(out, "i", attrs.kind === "linalg_vector_norm" ? "sqrt(acc)" : "acc")}`;
      break;
    }
    case "cat":
    case "stack": {
      const d = attrs.dim,
        inner = sizeOf(shape.slice(d + 1)),
        w = shape[d];
      let offset = 0;
      body = `let col=(i/${ui(inner)})%${ui(w)};`;
      for (const t of inputs) {
        const tw = kind === "stack" ? 1 : t.shape[d];
        body += `if(col>=${ui(offset)}&&col<${ui(offset + tw)}){${put(out, "i", get(t, `((i/${ui(w * inner)})*${ui(tw * inner)}+(col-${ui(offset)})*${ui(inner)}+i%${ui(inner)})`))}}`;
        offset += tw;
      }
      break;
    }
    case "embedding": {
      const w = a.shape[1];
      body = put(
        out,
        "i",
        get(a, `(u32(${get(b, `(i/${ui(w)})`)})*${ui(w)}+i%${ui(w)})`),
      );
      break;
    }
    case "scatter": {
      invocations = sizeOf(b.shape);
      const bs = strides(b.shape),
        os = strides(shape);
      const idx = shape
        .map((n, d) =>
          d === attrs.dim
            ? `u32(${get(b)})*${ui(os[d])}`
            : `((i/${ui(bs[d])})%${ui(b.shape[d])})*${ui(os[d])}`,
        )
        .join("+");
      body = put(out, `(${idx})`, fl(attrs.value));
      break;
    }
    case "repeat": {
      const os = strides(shape),
        is = strides(a.shape),
        shift = shape.length - a.shape.length;
      const idx =
        a.shape
          .map((n, d) => `((i/${ui(os[d + shift])})%${ui(n)})*${ui(is[d])}`)
          .join("+") || "0u";
      body = put(out, "i", get(a, `(${idx})`));
      break;
    }
    case "pad": {
      const os = strides(shape),
        is = strides(a.shape),
        p = attrs.pad;
      let valid = [],
        idx = [];
      for (let d = 0; d < shape.length; d++) {
        const before = p[2 * (shape.length - 1 - d)] ?? 0;
        body += `let q${d}=i32((i/${ui(os[d])})%${ui(shape[d])})-${before};`;
        valid.push(`q${d}>=0&&q${d}<${a.shape[d]}`);
        idx.push(`u32(q${d})*${ui(is[d])}`);
      }
      body += `var v=${fl(attrs.value)};if(${valid.join("&&")}){v=${get(a, `(${idx.join("+")})`)};}${put(out, "i", "v")}`;
      break;
    }
    case "cdist": {
      const m = a.shape.at(-2),
        n = b.shape.at(-2),
        k = a.shape.at(-1);
      body = `let h=i/${ui(m * n)};let r=(i/${ui(n)})%${ui(m)};let col=i%${ui(n)};var sum=0.0;
        ${loop("j", k, `let delta=${get(a, `((h*${ui(m)}+r)*${ui(k)}+j)`)}-${get(b, `((h*${ui(n)}+col)*${ui(k)}+j)`)};sum+=delta*delta;`)}${put(out, "i", "sqrt(sum)")}`;
      break;
    }
    case "svd": {
      invocations = sizeOf(a.shape) / 9;
      body = `var mat:array<f32,9>;var v:array<f32,9>;
        ${loop("j", 9, `mat[j]=${get(a, "(i*9u+j)")};v[j]=select(0.0,1.0,j/3u==j%3u);`)}
        for(var sweep=0u;sweep<16u;sweep++){for(var p=0u;p<2u;p++){for(var q=p+1u;q<3u;q++){
          var aa=0.0;var bb=0.0;var ab=0.0;
          for(var j=0u;j<3u;j++){aa+=mat[j*3u+p]*mat[j*3u+p];bb+=mat[j*3u+q]*mat[j*3u+q];ab+=mat[j*3u+p]*mat[j*3u+q];}
          if(abs(ab)<1e-7*sqrt(aa*bb)){continue;}
          let tau=(bb-aa)/(2.0*ab);let t=select(-1.0,1.0,tau>=0.0)/(abs(tau)+sqrt(1.0+tau*tau));let co=inverseSqrt(1.0+t*t);let sn=t*co;
          for(var j=0u;j<3u;j++){let x=mat[j*3u+p];let y=mat[j*3u+q];mat[j*3u+p]=co*x-sn*y;mat[j*3u+q]=sn*x+co*y;
            let vx=v[j*3u+p];let vy=v[j*3u+q];v[j*3u+p]=co*vx-sn*vy;v[j*3u+q]=sn*vx+co*vy;}
        }}}
        var norms:array<f32,3>;var order:array<u32,3>;
        for(var j=0u;j<3u;j++){norms[j]=sqrt(mat[j]*mat[j]+mat[3u+j]*mat[3u+j]+mat[6u+j]*mat[6u+j]);order[j]=j;}
        for(var p=0u;p<2u;p++){for(var q=p+1u;q<3u;q++){if(norms[order[q]]>norms[order[p]]){let tmp=order[p];order[p]=order[q];order[q]=tmp;}}}
        for(var j=0u;j<3u;j++){${put(outputs[1], "(i*3u+j)", "norms[order[j]]")}
          for(var k=0u;k<3u;k++){${put(outputs[0], "(i*9u+k*3u+j)", "mat[k*3u+order[j]]/max(norms[order[j]],1e-30)")}${put(outputs[2], "(i*9u+j*3u+k)", "v[k*3u+order[j]]")}}}`;
      break;
    }
    case "det": {
      body = `var mat:array<f32,9>;${loop("j", 9, `mat[j]=${get(a, "(i*9u+j)")};`)}${put(out, "i", "mat[0]*(mat[4]*mat[8]-mat[5]*mat[7])-mat[1]*(mat[3]*mat[8]-mat[5]*mat[6])+mat[2]*(mat[3]*mat[7]-mat[4]*mat[6])")}`;
      break;
    }
    default:
      throw Error(`Unsupported shader ${kind}`);
  }
  if (kind !== "matmul") {
    dispatch = [
      Math.min(65535, Math.ceil(invocations / 64)),
      Math.ceil(invocations / (64 * 65535)),
      1,
    ];
    body =
      `let i=gid.x+gid.y*4194240u;if(i>=${ui(invocations)}){return;}` + body;
  }
  const header = bindings
    .map(
      (b, i) =>
        `@group(0) @binding(${i}) var<storage,${b.write ? "read_write" : "read"}> ${b.name}:array<${b.storage.encoding ? "u32" : "f32"}>;`,
    )
    .join("\n");
  const used = bindings
    .map((b, binding) => ({ ...b, binding }))
    .filter((b) => new RegExp("\\b" + b.name + "\\[").test(body));
  return {
    code: `${header}\n${extra}\n@compute @workgroup_size(${workgroup}) fn main(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>,@builtin(workgroup_id) wg:vec3<u32>){${body}}`,
    bindings: used,
    dispatch,
  };
}
