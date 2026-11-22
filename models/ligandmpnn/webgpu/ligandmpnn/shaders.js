/** Tiled matmul, FP32 accumulation for all paths. Optional f16 is weight storage. */
export function linearShader(
  R,
  I,
  O,
  { packed = false, f16 = false, bias = true } = {},
) {
  const load = packed
    ? f16
      ? "f32(bitcast<vec2<f16>>(w[index/2u])[index%2u])"
      : "unpack2x16float(w[index/2u])[index%2u]"
    : "w[index]";
  return `${packed && f16 ? "enable f16;" : ""}
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> w:array<${packed ? "u32" : "f32"}>;
${bias ? "@group(0) @binding(2) var<storage,read> b:array<f32>;" : ""}
@group(0) @binding(${bias ? 3 : 2}) var<storage,read_write> out:array<f32>;
var<workgroup> tileA:array<f32,256>;
var<workgroup> tileB:array<f32,256>;
@compute @workgroup_size(16,16)
fn main(@builtin(local_invocation_id) lid:vec3<u32>,@builtin(workgroup_id) gid:vec3<u32>){
 let row=gid.y*16u+lid.y;let col=gid.x*16u+lid.x;var acc=0.0;
 for(var base=0u;base<${I}u;base+=16u){
  var av=0.0;var bv=0.0;
  if(row<${R}u && base+lid.x<${I}u){av=x[row*${I}u+base+lid.x];}
  if(col<${O}u && base+lid.y<${I}u){let index=col*${I}u+base+lid.y;bv=${load};}
  tileA[lid.y*16u+lid.x]=av;tileB[lid.y*16u+lid.x]=bv;workgroupBarrier();
  for(var k=0u;k<16u;k++){acc+=tileA[lid.y*16u+k]*tileB[k*16u+lid.x];}
  workgroupBarrier();
 }
 if(row<${R}u&&col<${O}u){out[row*${O}u+col]=acc${bias ? "+b[col]" : ""};}
}`;
}
export function elementShader(bindings, body, N) {
  return (
    bindings
      .map(
        (b, i) =>
          `@group(0) @binding(${i}) var<storage,${i === bindings.length - 1 ? "read_write" : "read"}> ${b}:array<f32>;`,
      )
      .join("\n") +
    `\n@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x+gid.y*4194240u;if(i>=${N}u){return;}${body}}`
  );
}
export const GELU_BODY =
  "let v=x[i];let z=abs(v)*0.7071067811865476;let t=1.0/(1.0+0.3275911*z);let e=1.0-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*exp(-z*z);out[i]=0.5*v*(1.0+sign(v)*e);";
export function normShader(R, D) {
  return `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
var<workgroup> partial: array<f32, 128>;

fn rowSum(value: f32, lane: u32) -> f32 {
  partial[lane] = value;
  workgroupBarrier();
  for (var stride = 64u; stride > 0u; stride /= 2u) {
    if (lane < stride) { partial[lane] += partial[lane + stride]; }
    workgroupBarrier();
  }
  let total = partial[0];
  workgroupBarrier(); // All lanes must read the result before the next reduction reuses partial.
  return total;
}

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lane: u32, @builtin(workgroup_id) group: vec3<u32>) {
  let row = group.x + group.y * 65535u;
  if (row >= ${R}u) { return; }
  let base = row * ${D}u;
  let origin = x[base];
  var sum = 0.0;
  for (var col = lane; col < ${D}u; col += 128u) { sum += x[base + col] - origin; }
  let mean = origin + rowSum(sum, lane) / f32(${D}u);
  var squares = 0.0;
  for (var col = lane; col < ${D}u; col += 128u) {
    let centered = x[base + col] - mean;
    squares += centered * centered;
  }
  let invStd = inverseSqrt(rowSum(squares, lane) / f32(${D}u) + 0.00001);
  for (var col = lane; col < ${D}u; col += 128u) {
    out[base + col] = (x[base + col] - mean) * invStd * w[col] + b[col];
  }
}`;
}
