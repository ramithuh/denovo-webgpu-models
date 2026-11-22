import constants from "./constants.js";
export function normLast(o, x, clamp = 0) {
  const shape = x.shape.slice(0, -1),
    n = x.shape.at(-1);
  return o.kernel(
    [...shape, 1],
    [x],
    `const b=i*${n};let v=0;for(let j=0;j<${n};j++)v=rfFma(a0[b+j],a0[b+j],v);o[i]=Math.max(Math.sqrt(v),${clamp});`,
    `let b=i*${n}u;var v=0.0;for(var j=0u;j<${n}u;j++){v=fma(a0[b+j],a0[b+j],v);}o[i]=max(sqrt(v),${clamp.toExponential()});`,
  );
}
export function cross(o, a, b) {
  const shape = a.shape;
  const preJS = "const k=i%3,base=i-k,j=(k+1)%3,l=(k+2)%3;",
    preWG = "let k=i%3u;let base=i-k;let j=(k+1u)%3u;let l=(k+2u)%3u;";
  const expr = "o[i]=a0[base+j]*a1[base+l]-a0[base+l]*a1[base+j];";
  return o.kernel(
    shape,
    [a, b],
    preJS +
      "o[i]=rfFma(a0[base+j],a1[base+l],-Math.fround(a0[base+l]*a1[base+j]));",
    preWG + "o[i]=fma(a0[base+j],a1[base+l],-a0[base+l]*a1[base+j]);",
  );
}
export function cbeta(o, xyz) {
  const L = xyz.shape[0],
    a = (k) => o.reshape(o.slice(xyz, 1, k, k + 1), [L, 3]),
    N = a(0),
    CA = a(1),
    C = a(2),
    b = o.sub(CA, N),
    c = o.sub(C, CA);
  return o.add(
    o.sub(
      o.add(o.mul(cross(o, b, c), -0.58273431), o.mul(b, 0.56802827)),
      o.mul(c, 0.54067466),
    ),
    CA,
  );
}
export function cdist(o, x) {
  const L = x.shape[0];
  if (L > 25) {
    const sq = o.mul(x, x),
      norm = o.kernel(
        [L],
        [sq],
        `o[i]=Math.fround(Math.fround(a0[i*3]+a0[i*3+1])+a0[i*3+2]);`,
        `o[i]=(a0[i*3u]+a0[i*3u+1u])+a0[i*3u+2u];`,
      );
    return o.kernel(
      [L, L],
      [x, norm],
      `const row=Math.floor(i/${L}),col=i%${L};let s=0;for(let k=0;k<3;k++)s=rfFma(-2*a0[row*3+k],a0[col*3+k],s);s=Math.fround(s+a1[row]);s=Math.fround(s+a1[col]);o[i]=Math.sqrt(Math.max(0,s));`,
      `let row=i/${L}u;let col=i%${L}u;var s=0.0;for(var k=0u;k<3u;k++){s=fma(-2.0*a0[row*3u+k],a0[col*3u+k],s);}s=s+a1[row];s=s+a1[col];o[i]=sqrt(max(0.0,s));`,
    );
  }
  return o.kernel(
    [L, L],
    [x],
    `const a=Math.floor(i/${L})*3,b=i%${L}*3;let s=0;for(let k=0;k<3;k++){const d=Math.fround(a0[a+k]-a0[b+k]);s=rfFma(d,d,s);}o[i]=Math.sqrt(s);`,
    `let a=(i/${L}u)*3u;let b=(i%${L}u)*3u;var s=0.0;for(var k=0u;k<3u;k++){let d=a0[a+k]-a0[b+k];s+=d*d;}o[i]=sqrt(s);`,
  );
}
export function rbf(o, dist) {
  const mu = o.tensor([36], constants.rbfMu),
    q = o.div(o.sub(o.reshape(dist, [...dist.shape, 1]), mu), 20 / 36);
  return o.unary(o.mul(q, q), "exp(-$0)");
}
export function quaternionRotation(o, rotation) {
  const L = rotation.shape[0],
    square = o.mul(rotation, rotation),
    n = o.kernel(
      [L],
      [square],
      `const b=i*3;const sum=Math.fround(Math.fround(a0[b]+a0[b+1])+a0[b+2]);o[i]=Math.sqrt(Math.fround(1+sum));`,
      `let b=i*3u;let sum=(a0[b]+a0[b+1u])+a0[b+2u];o[i]=sqrt(1.0+sum);`,
    ),
    q = o.div(
      o.cat([o.constant([L, 1], 1), rotation], 1),
      o.reshape(n, [L, 1]),
    ),
    qq = o.einsum("ni,nj->nij", q, q),
    diagAB = o.kernel(
      [L, 3],
      [qq],
      `const b=Math.floor(i/3)*16;o[i]=i%3===0?a0[b]+a0[b+5]:a0[b]-a0[b+5];`,
      `let b=(i/3u)*16u;o[i]=select(a0[b]-a0[b+5u],a0[b]+a0[b+5u],i%3u==0u);`,
    ),
    diagABC = o.kernel(
      [L, 3],
      [diagAB, qq],
      `const b=Math.floor(i/3)*16;o[i]=i%3===1?a0[i]+a1[b+10]:a0[i]-a1[b+10];`,
      `let b=(i/3u)*16u;o[i]=select(a0[i]-a1[b+10u],a0[i]+a1[b+10u],i%3u==1u);`,
    );
  const exprs = [
    "a1[Math.floor(i/9)*3]-a0[b+15]",
    "2*a0[b+6]-2*a0[b+3]",
    "2*a0[b+7]+2*a0[b+2]",
    "2*a0[b+6]+2*a0[b+3]",
    "a1[Math.floor(i/9)*3+1]-a0[b+15]",
    "2*a0[b+11]-2*a0[b+1]",
    "2*a0[b+7]-2*a0[b+2]",
    "2*a0[b+11]+2*a0[b+1]",
    "a1[Math.floor(i/9)*3+2]+a0[b+15]",
  ];
  return o.kernel(
    [L, 3, 3],
    [qq, diagABC],
    `const b=Math.floor(i/9)*16,k=i%9;const v=[${exprs.join(",")}];o[i]=v[k];`,
    `let b=(i/9u)*16u;let k=i%9u;var v=array<f32,9>(${exprs.join(",").replaceAll("Math.floor(i/9)", "(i/9u)")});o[i]=v[k];`,
  );
}

export function sphericalHarmonics(o, relative) {
  const E = relative.shape[0],
    unit = o.div(relative, normLast(o, relative, 1e-12)),
    x = o.slice(unit, 1, 0, 1),
    y = o.slice(unit, 1, 1, 2),
    z = o.slice(unit, 1, 2, 3),
    x2 = o.mul(x, x),
    z2 = o.mul(z, z);
  const s0 = o.constant([E, 1], Math.sqrt(1) / Math.sqrt(4 * Math.PI)),
    s1 = o.mul(unit, Math.sqrt(3) / Math.sqrt(4 * Math.PI)),
    s2 = o.mul(
      o.cat(
        [
          o.mul(o.mul(x, Math.sqrt(3)), z),
          o.mul(o.mul(x, Math.sqrt(3)), y),
          o.sub(o.mul(y, y), o.mul(o.add(x2, z2), 0.5)),
          o.mul(o.mul(y, Math.sqrt(3)), z),
          o.mul(o.sub(z2, x2), Math.sqrt(3) / 2),
        ],
        1,
      ),
      Math.sqrt(5) / Math.sqrt(4 * Math.PI),
    );
  return o.cat([s0, s1, s2], 1);
}
export function rigidFrames(o, xyz) {
  const L = xyz.shape[0],
    at = (k) => o.reshape(o.slice(xyz, 1, k, k + 1), [L, 3]),
    CA = at(1),
    v1 = o.sub(at(2), CA),
    v2 = o.sub(at(0), CA),
    e1 = o.div(v1, o.add(normLast(o, v1), 1e-8)),
    u2 = o.sub(v2, o.mul(o.reshape(o.einsum("ni,ni->n", e1, v2), [L, 1]), e1)),
    e2 = o.div(u2, o.add(normLast(o, u2), 1e-8)),
    e3 = cross(o, e1, e2);
  return {
    R: o.cat(
      [
        o.reshape(e1, [L, 3, 1]),
        o.reshape(e2, [L, 3, 1]),
        o.reshape(e3, [L, 3, 1]),
      ],
      2,
    ),
    T: CA,
  };
}
