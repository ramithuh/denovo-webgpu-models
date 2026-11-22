import { sizeOf } from "../shared/tensor.js";
export function linearGPU(o, x, w, b) {
  const K = x.shape.at(-1),
    N = w.shape[0],
    M = x.size / K;
  if (w.shape[1] !== K) throw Error("Linear shape mismatch");
  const out = o.alloc([...x.shape.slice(0, -1), N]),
    raw = w.dtype && w.dtype !== "F32";
  const load =
    w.dtype === "F16"
      ? "unpack2x16float(W[i/2u])[i%2u]"
      : w.dtype === "I8"
        ? `f32(i32((W[i/4u]>>((i%4u)*8u))<<24u)>>24)*bitcast<f32>(W[${w.scaleOffset}u+i/${K}u])`
        : "W[i]";
  const code = `@group(0) @binding(0) var<storage,read> X:array<f32>;@group(0) @binding(1) var<storage,read> W:array<${raw ? "u32" : "f32"}>;${b ? "@group(0) @binding(2) var<storage,read> B:array<f32>;" : ""}@group(0) @binding(${b ? 3 : 2}) var<storage,read_write> O:array<f32>;
 var<workgroup> A:array<f32,256>;var<workgroup> T:array<f32,256>;fn weight(i:u32)->f32{return ${load};}
 @compute @workgroup_size(16,16) fn main(@builtin(local_invocation_id) l:vec3<u32>,@builtin(workgroup_id) g:vec3<u32>){let row=g.y*16u+l.y;let col=g.x*16u+l.x;var sum=0.0;${b ? `if(col<${N}u){sum=B[col];}` : ""}
 for(var base=0u;base<${K}u;base+=16u){var a=0.0;var w=0.0;if(row<${M}u&&base+l.x<${K}u){a=X[row*${K}u+base+l.x];}if(col<${N}u&&base+l.y<${K}u){w=weight(col*${K}u+base+l.y);}A[l.y*16u+l.x]=a;T[l.y*16u+l.x]=w;workgroupBarrier();for(var j=0u;j<16u;j++){sum=fma(A[l.y*16u+j],T[j*16u+l.x],sum);}workgroupBarrier();}
 if(row<${M}u&&col<${N}u){O[row*${N}u+col]=sum;}}`;
  let pipeline = o.pipelines.get(code);
  if (!pipeline) {
    pipeline = o.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: o.device.createShaderModule({ code }),
        entryPoint: "main",
      },
    });
    o.pipelines.set(code, pipeline);
  }
  const inputs = b ? [x, w, b, out] : [x, w, out],
    bg = o.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: inputs.map((t, binding) => ({
        binding,
        resource: { buffer: t.buffer },
      })),
    });
  o.encoder ??= o.device.createCommandEncoder();
  const pass = o.encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(N / 16), Math.ceil(M / 16));
  pass.end();
  if (++o.dispatches % 128 === 0) o.flush();
  return out;
}
