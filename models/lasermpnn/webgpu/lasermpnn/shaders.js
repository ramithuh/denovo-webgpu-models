// All learned transformations and graph reductions execute on the GPU.
const header = `
struct Params { r:u32, c:u32, k:u32, d:u32, e:u32, f:u32, g:u32, h:u32 };
@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read> b:array<f32>;
@group(0) @binding(2) var<storage,read> c:array<f32>;
@group(0) @binding(3) var<storage,read> d:array<f32>;
@group(0) @binding(4) var<storage,read_write> out:array<f32>;
@group(0) @binding(5) var<uniform> p:Params;
fn sigmoid(x:f32)->f32 { return 1.0/(1.0+exp(-x)); }
fn gelu(x:f32)->f32 {
 let z=abs(x)*0.7071067811865475; let t=1.0/(1.0+0.3275911*z);
 let erf=1.0-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*exp(-z*z);
 return 0.5*x*(1.0+select(erf,-erf,x<0.0));
}
`;
const bodies = {
  act: `out[i]=select(gelu(a[i]),sigmoid(a[i]),p.k==1u);`,
  add: `out[i]=a[i]+b[i];`,
  cat: `if(col<p.k){out[i]=a[row*p.k+col];}else{out[i]=b[row*(p.c-p.k)+col-p.k];}`,
  gather: `out[i]=a[u32(b[row])*p.c+col];`,
  vecLinear: `let channel=col/3u;let axis=col%3u;var sum=0.0;for(var j=0u;j<p.k;j++){sum+=a[row*p.k*3u+j*3u+axis]*b[channel*p.k+j];}out[i]=sum;`,
  vecLength: `out[i]=sqrt(max(0.00000001,a[3u*i]*a[3u*i]+a[3u*i+1u]*a[3u*i+1u]+a[3u*i+2u]*a[3u*i+2u]));`,
  vecScale: `out[i]=a[i]*b[i/3u];`,
  vecNorm: `var sum=0.0;for(var j=0u;j<p.c;j+=3u){let at=row*p.c+j;sum+=max(0.00000001,a[at]*a[at]+a[at+1u]*a[at+1u]+a[at+2u]*a[at+2u]);}out[i]=a[i]*inverseSqrt(sum/f32(p.c/3u));`,
  rbf: `let diff=(a[i/p.k]-b[i%p.k])/bitcast<f32>(p.d);out[i]=exp(-diff*diff)+0.0001;`,
  dots: `let av=col/p.d;let bv=col%p.d;var sum=0.0;var aa=0.0;var bb=0.0;for(var j=0u;j<3u;j++){let x=a[row*p.k*3u+av*3u+j];let y=b[row*p.d*3u+bv*3u+j];sum+=x*y;aa+=x*x;bb+=y*y;}out[i]=sum*inverseSqrt(max(aa,0.00000001)*max(bb,0.00000001));`,
  atten: `var sum=0.0;for(var j=0u;j<p.k;j++){let v=a[row*p.c*p.k+col*p.k+j];sum+=select(0.2*v,v,v>=0.0)*b[col*p.k+j];}out[i]=sum;`,
  pool: `var sum=0.0;let begin=u32(d[row]);let end=u32(d[row+1u]);for(var head=0u;head<p.d;head++){
 var maximum=-3.402823466e38;for(var e=begin;e<end;e++){let edge=u32(c[e]);maximum=max(maximum,b[edge*p.d+head]);}
 var denom=0.0;var accum=0.0;for(var e=begin;e<end;e++){let edge=u32(c[e]);let prob=exp(b[edge*p.d+head]-maximum);denom+=prob;let at=edge*p.k+select(head*p.c,0u,p.k==p.c)+col;accum+=prob*a[at];}
 if(denom>0.0){sum+=accum/denom/f32(p.d);}}out[i]=sum;`,
  catOneHot: `if(col<p.k){out[i]=a[col];}else{out[i]=select(0.0,1.0,(col-p.k)==u32(b[0]));}`,
  chiAngle: `let raw=-180.0+f32(u32(a[0]))*5.0+b[0];let s=raw+180.0;let m=s-360.0*floor(s/360.0);out[0]=select(m,0.0,m>=360.0)-180.0;`,
  maskRow: `out[i]=a[i]*b[p.k];`,
};
export const OP_SHADERS = Object.fromEntries(
  Object.entries(bodies).map(([kind, body]) => [
    kind,
    header +
      `
@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x+id.y*8388480u;if(i>=p.r*p.c){return;}let row=i/p.c;let col=i%p.c;${body}
}`,
  ]),
);
OP_SHADERS.sampleIndex = header + `
@compute @workgroup_size(1)
fn main() {
  let k = p.k;
  let temp = bitcast<f32>(p.d);
  let uniformVal = bitcast<f32>(p.e);
  let hasMask = p.f != 0u;
  if (temp <= 0.0) {
    var bestIdx = 0u;
    var bestVal = -3.402823466e38;
    var hasBest = false;
    for (var j = 0u; j < k; j++) {
      if (!hasMask || b[j] > 0.5) {
        if (!hasBest || a[j] > bestVal) { bestVal = a[j]; bestIdx = j; hasBest = true; }
      }
    }
    out[0] = f32(bestIdx);
    return;
  }
  var maxVal = -3.402823466e38;
  var lastValid = k - 1u;
  var hasValid = false;
  for (var j = 0u; j < k; j++) {
    if (!hasMask || b[j] > 0.5) {
      if (!hasValid || a[j] > maxVal) { maxVal = a[j]; }
      lastValid = j;
      hasValid = true;
    }
  }
  var sum = 0.0;
  for (var j = 0u; j < k; j++) {
    if (!hasMask || b[j] > 0.5) {
      sum += exp((a[j] - maxVal) / temp);
    }
  }
  var cdf = 0.0;
  var choice = lastValid;
  for (var j = 0u; j < k; j++) {
    if (!hasMask || b[j] > 0.5) {
      cdf += exp((a[j] - maxVal) / temp) / sum;
      if (uniformVal <= cdf) { choice = j; break; }
    }
  }
  out[0] = f32(choice);
}`;
OP_SHADERS.chiRbf = header + `
@compute @workgroup_size(1)
fn main() {
  let ang = a[0];
  var s = 0.0;
  for (var j = 0u; j < 72u; j++) {
    let center = -180.0 + f32(j) * 5.0;
    let d1raw = center - ang;
    let d1 = d1raw - 360.0 * floor(d1raw / 360.0);
    let d2raw = ang - center;
    let d2 = d2raw - 360.0 * floor(d2raw / 360.0);
    let mn = min(d1, d2) / 2.5;
    s += exp(-0.5 * mn * mn);
  }
  for (var j = 0u; j < 72u; j++) {
    let center = -180.0 + f32(j) * 5.0;
    let d1raw = center - ang;
    let d1 = d1raw - 360.0 * floor(d1raw / 360.0);
    let d2raw = ang - center;
    let d2 = d2raw - 360.0 * floor(d2raw / 360.0);
    let mn = min(d1, d2) / 2.5;
    out[j] = exp(-0.5 * mn * mn) / s;
  }
}`;
OP_SHADERS.norm = header + `
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
  if (row >= p.r) { return; }
  let base = row * p.c;
  var sum = 0.0;
  for (var col = lane; col < p.c; col += 128u) { sum += a[base + col]; }
  let mean = rowSum(sum, lane) / f32(p.c);
  var squares = 0.0;
  for (var col = lane; col < p.c; col += 128u) {
    let centered = a[base + col] - mean;
    squares += centered * centered;
  }
  let invStd = inverseSqrt(rowSum(squares, lane) / f32(p.c) + 0.00001);
  for (var col = lane; col < p.c; col += 128u) {
    out[base + col] = (a[base + col] - mean) * invStd * b[col] + c[col];
  }
}`;

export function gemmSource(f16 = false) {
  return `${f16 ? "enable f16;" : ""}
struct Params { r:u32,c:u32,k:u32,hasBias:u32 };
@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read> b:array<${f16 ? "f16" : "f32"}>;
@group(0) @binding(2) var<storage,read> bias:array<f32>;
@group(0) @binding(3) var<storage,read_write> out:array<f32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> tileA:array<array<f32,16>,16>;
var<workgroup> tileB:array<array<f32,16>,16>;
@compute @workgroup_size(16,16)
fn main(@builtin(local_invocation_id) local:vec3<u32>,@builtin(workgroup_id) group:vec3<u32>){
 let row=group.y*16u+local.y;let col=group.x*16u+local.x;var sum=0.0;
 for(var offset=0u;offset<p.k;offset+=16u){
  tileA[local.y][local.x]=0.0;tileB[local.y][local.x]=0.0;
  if(row<p.r&&offset+local.x<p.k){tileA[local.y][local.x]=a[row*p.k+offset+local.x];}
  if(group.x*16u+local.y<p.c&&offset+local.x<p.k){tileB[local.y][local.x]=f32(b[(group.x*16u+local.y)*p.k+offset+local.x]);}
  workgroupBarrier();
  for(var j=0u;j<16u;j++){sum+=tileA[local.y][j]*tileB[local.x][j];}
  workgroupBarrier();
 }
 if(row<p.r&&col<p.c){if(p.hasBias!=0u){sum+=bias[col];}out[row*p.c+col]=sum;}
}`;
}
