import { PERIODIC_TABLE } from "../../shared/ligandmpnn/constants.js";
import { elementShader } from "./shaders.js";
const RBF = `fn rbf(d:f32,k:u32)->f32{var mu=2.0+f32(k)*(20.0/15.0);if(k>7u){mu=22.0-f32(15u-k)*(20.0/15.0);}let z=(d-mu)/1.25;return exp(-z*z);}`;
const XYZ = `fn atom(r:u32,a:u32)->vec3<f32>{if(a==4u){return vec3<f32>(CB[r*3u],CB[r*3u+1u],CB[r*3u+2u]);}let i=r*12u+a*3u;return vec3<f32>(X[i],X[i+1u],X[i+2u]);}`;
const YXYZ = `fn context(i:u32)->vec3<f32>{return vec3<f32>(Y[i*3u],Y[i*3u+1u],Y[i*3u+2u]);}`;
/** Geometric RBFs, residue frames, atom types and relative positions on WebGPU. */
export function gpuRawFeatures(ops, p) {
  const { L, K, M } = p,
    t = (a, shape) => ops.tensor(a, shape),
    X = t(p.X, [L, 12]),
    CB = t(p.CB, [L, 3]),
    Y = t(p.Y, [L * M, 3]),
    idx = t(p.E_idx, [L * K, 1]),
    res = t(p.R_idx, [L, 1]),
    chains = t(p.chain_labels, [L, 1]),
    types = t(p.Y_t, [L * M, 1]),
    mask = t(p.mask, [L, 1]);
  const kernel = (names, inputs, shape, helpers, body) => {
    const n = shape[0] * shape[1],
      code = helpers + "\n" + elementShader([...names, "out"], body, n);
    return ops.dispatch(code, inputs, ops.zeros(shape), [
      Math.min(65535, Math.ceil(n / 64)),
      Math.ceil(n / 4194240),
    ]);
  };
  const positional = kernel(
    ["idx", "res", "chains"],
    [idx, res, chains],
    [L * K, 66],
    "",
    `let row=i/66u;let a=row/${K}u;let b=u32(idx[row]);var d=65u;if(chains[a]==chains[b]){d=u32(clamp(res[a]-res[b]+32.0,0.0,64.0));}out[i]=select(0.0,1.0,i%66u==d);`,
  );
  const pairs = [
    [1, 1],
    [0, 0],
    [2, 2],
    [3, 3],
    [4, 4],
    [1, 0],
    [1, 2],
    [1, 3],
    [1, 4],
    [0, 2],
    [0, 3],
    [0, 4],
    [4, 2],
    [4, 3],
    [3, 2],
    [0, 1],
    [2, 1],
    [3, 1],
    [4, 1],
    [2, 0],
    [3, 0],
    [4, 0],
    [2, 4],
    [3, 4],
    [2, 3],
  ];
  const pairConst = `const pairs=array<vec2<u32>,25>(${pairs.map(([a, b]) => `vec2<u32>(${a}u,${b}u)`).join(",")});`;
  const rbf = kernel(
    ["X", "CB", "idx", "mask"],
    [X, CB, idx, mask],
    [L * K, 400],
    XYZ + RBF + pairConst,
    `let row=i/400u;let a=row/${K}u;let b=u32(idx[row]);let pair=(i%400u)/16u;let d=atom(a,pairs[pair].x)-atom(b,pairs[pair].y);var dist=sqrt(dot(d,d)+0.000001);if(pair==0u&&mask[a]*mask[b]==0.0){dist=0.0;for(var j=0u;j<${L}u;j++){let v=atom(a,1u)-atom(j,1u);dist=max(dist,mask[a]*mask[j]*sqrt(dot(v,v)+0.000001));}}out[i]=rbf(dist,i%16u);`,
  );
  const typeHelpers = `const groups=array<u32,119>(${PERIODIC_TABLE[1].map((x) => x + "u").join(",")});const periods=array<u32,119>(${PERIODIC_TABLE[2].map((x) => x + "u").join(",")});`;
  const typeFeatures = kernel(
    ["types"],
    [types],
    [L * M, 147],
    typeHelpers,
    `let row=i/147u;let col=i%147u;let t=u32(types[row]);out[i]=select(0.0,1.0,col==t||col==120u+groups[t]||col==139u+periods[t]);`,
  );
  const distances = kernel(
    ["X", "CB", "Y"],
    [X, CB, Y],
    [L * M, 80],
    XYZ + YXYZ + RBF,
    `let row=i/80u;let d=atom(row/${M}u,(i%80u)/16u)-context(row);out[i]=rbf(sqrt(dot(d,d)+0.000001),i%16u);`,
  );
  const angles = kernel(
    ["X", "CB", "Y"],
    [X, CB, Y],
    [L * M, 4],
    XYZ + YXYZ,
    `let row=i/4u;let r=row/${M}u;let ca=atom(r,1u);let v1=atom(r,0u)-ca;let v2=atom(r,2u)-ca;let e1=v1/max(length(v1),1e-12);let u2=v2-e1*dot(e1,v2);let e2=u2/max(length(u2),1e-12);let e3=cross(e1,e2);let d=context(row)-ca;let v=vec3<f32>(dot(e1,d),dot(e2,d),dot(e3,d));let xy=sqrt(v.x*v.x+v.y*v.y+1e-8);let xyz=length(v)+1e-8;let features=vec4<f32>(v.x/xy,v.y/xy,xy/xyz,v.z/xyz);out[i]=features[i%4u];`,
  );
  const yRbf = kernel(
    ["Y"],
    [Y],
    [L * M * M, 16],
    YXYZ + RBF,
    `let row=i/16u;let a=row/${M}u;let b=(row/${M * M}u)*${M}u+row%${M}u;let d=context(a)-context(b);out[i]=rbf(sqrt(dot(d,d)+0.000001),i%16u);`,
  );
  return { positional, rbf, types: typeFeatures, distances, angles, yRbf };
}
