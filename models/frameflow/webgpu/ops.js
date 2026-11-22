import { Ops } from "../shared/ops.js";
const product = (s) => s.reduce((a, b) => a * b, 1);
const params = `struct Params{v:array<vec4<f32>,16>};@group(0) @binding(0) var<uniform> params:Params;fn pf(i:u32)->f32{return params.v[i/4u][i%4u];}fn u(i:u32)->u32{return u32(pf(i));}`;
const quat = `fn qmat(q:vec4<f32>)->mat3x3<f32>{let w=q.x;let x=q.y;let y=q.z;let z=q.w;return mat3x3<f32>(vec3<f32>(1.0-2.0*(y*y+z*z),2.0*(x*y+w*z),2.0*(x*z-w*y)),vec3<f32>(2.0*(x*y-w*z),1.0-2.0*(x*x+z*z),2.0*(y*z+w*x)),vec3<f32>(2.0*(x*z+w*y),2.0*(y*z-w*x),1.0-2.0*(x*x+y*y)));}`;
const matquat = `fn matquat(i:u32)->vec4<f32>{let r0=a[i];let r1=a[i+1u];let r2=a[i+2u];let r3=a[i+3u];let r4=a[i+4u];let r5=a[i+5u];let r6=a[i+6u];let r7=a[i+7u];let r8=a[i+8u];let d=vec4<f32>(1.0+r0+r4+r8,1.0+r0-r4-r8,1.0-r0+r4-r8,1.0-r0-r4+r8);var k=0u;for(var j=1u;j<4u;j++){if(d[j]>d[k]){k=j;}}let v=2.0*sqrt(max(0.0,d[k]));var q=vec4<f32>(0.0);if(k==0u){q=vec4<f32>(v/4.0,(r7-r5)/v,(r2-r6)/v,(r3-r1)/v);}else if(k==1u){q=vec4<f32>((r7-r5)/v,v/4.0,(r1+r3)/v,(r2+r6)/v);}else if(k==2u){q=vec4<f32>((r2-r6)/v,(r1+r3)/v,v/4.0,(r5+r7)/v);}else{q=vec4<f32>((r3-r1)/v,(r2+r6)/v,(r5+r7)/v,v/4.0);}return normalize(q);}`;
export class GPUOps extends Ops {
  constructor(device, { useF16 = false } = {}) {
    super({});
    this.nativeF16 = useF16 && device.features.has("shader-f16");
    this.device = device;
    this.weights = {};
    this.live = new Set();
    this.pool = [];
    this.pipelines = new Map();
    this.weightBytes = 0;
    this.scratchBytes = 0;
    this.peakBytes = 0;
    this.destroyed = false;
    this.encoder = null;
    this.dispatches = 0;
    this.parameterData = new Float32Array(64 * 4096);
    this.parameterBuffer = this.buffer(
      this.parameterData.byteLength,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    this.scratchBytes += this.parameterData.byteLength;
  }
  buffer(bytes, usage) {
    const n = Math.max(4, Math.ceil(bytes / 4) * 4);
    if (
      n > this.device.limits.maxBufferSize ||
      (usage & GPUBufferUsage.STORAGE &&
        n > this.device.limits.maxStorageBufferBindingSize)
    )
      throw new RangeError(`Buffer exceeds device limit: ${n}`);
    return this.device.createBuffer({ size: n, usage });
  }
  uploadWeight(name, t) {
    const buffer = this.buffer(
      t.bytes.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    );
    this.device.queue.writeBuffer(buffer, 0, t.bytes);
    this.weightBytes += buffer.size;
    const w = { buffer, shape: t.shape, dtype: t.dtype };
    this.weights[name] = w;
    if (t.scales) {
      w.scales = this.buffer(
        t.scales.byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      );
      this.device.queue.writeBuffer(w.scales, 0, t.scales);
      this.weightBytes += w.scales.size;
    }
  }
  alloc(shape) {
    const bytes = Math.max(4, product(shape) * 4);
    let i = this.pool.findIndex((t) => t.buffer.size === bytes);
    let t;
    if (i >= 0) {
      t = this.pool.splice(i, 1)[0];
      t.shape = shape;
    } else {
      t = {
        buffer: this.buffer(
          bytes,
          GPUBufferUsage.STORAGE |
            GPUBufferUsage.COPY_DST |
            GPUBufferUsage.COPY_SRC,
        ),
        shape,
      };
      this.scratchBytes += t.buffer.size;
      this.peakBytes = Math.max(
        this.peakBytes,
        this.scratchBytes + this.weightBytes,
      );
    }
    this.live.add(t);
    return t;
  }
  tensor(data, shape) {
    const t = this.alloc(shape);
    this.device.queue.writeBuffer(
      t.buffer,
      0,
      data instanceof Float32Array ? data : Float32Array.from(data),
    );
    return t;
  }
  collect(keep = []) {
    const set = new Set(keep);
    for (const t of this.live)
      if (!set.has(t)) {
        this.live.delete(t);
        this.pool.push(t);
      }
  }
  begin() {
    if (this.destroyed) throw Error("Engine destroyed");
    this.encoder = this.device.createCommandEncoder();
    this.dispatches = 0;
  }
  run(name, inputs, shape, p, body) {
    const out = this.alloc(shape),
      n = product(shape);
    if (!this.encoder) throw Error("No active GPU operation");
    if (this.dispatches >= 4096) throw Error("Dispatch capacity exceeded");
    const binding =
      inputs
        .map(
          (_, i) =>
            `@group(0) @binding(${i + 1}) var<storage,read> ${String.fromCharCode(97 + i)}:array<f32>;`,
        )
        .join("\n") +
      `\n@group(0) @binding(${inputs.length + 1}) var<storage,read_write> o:array<f32>;`;
    let source =
      params +
      "\n" +
      binding +
      "\n" +
      quat +
      (name === "frames" ? matquat : "") +
      `\n@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>){let idx=id.x+id.y*groups.x*64u;if(idx>=u(0)){return;}${body.replace(/\bf\(/g, "pf(")}}`;
    if (name === "linear") source = this.linearSource(inputs[1].dtype);
    const key =
      name +
      (name === "linear"
        ? inputs[1].dtype +
          (inputs[1].dtype === "F16" && this.nativeF16 ? "native" : "")
        : "");
    let pipeline = this.pipelines.get(key);
    if (!pipeline) {
      const module = this.device.createShaderModule({
        label: key,
        code: source,
      });
      pipeline = this.device.createComputePipeline({
        label: key,
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
      this.pipelines.set(key, pipeline);
    }
    const slot = this.dispatches++;
    this.parameterData.set([n, ...p], slot * 64);
    let buffers = inputs.map((t) => t.buffer);
    if (name === "linear") buffers.push(inputs[1].scales ?? inputs[2].buffer);
    const entries = [
      {
        binding: 0,
        resource: {
          buffer: this.parameterBuffer,
          offset: slot * 256,
          size: 256,
        },
      },
      ...buffers.map((buffer, i) => ({ binding: i + 1, resource: { buffer } })),
      { binding: buffers.length + 1, resource: { buffer: out.buffer } },
    ];
    const filtered =
      name === "linear" && inputs[1].dtype !== "I8"
        ? entries.filter((e) => e.binding !== 4)
        : entries;
    const bind = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: filtered,
    });
    const pass = this.encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    if (name === "linear")
      pass.dispatchWorkgroups(Math.ceil(p[2] / 8), Math.ceil(p[0] / 8));
    else {
      const groups = Math.ceil(n / 64),
        width = Math.min(
          groups,
          this.device.limits.maxComputeWorkgroupsPerDimension,
        );
      pass.dispatchWorkgroups(width, Math.ceil(groups / width));
    }
    pass.end();
    return out;
  }
  linearSource(dtype) {
    const read =
      dtype === "F32"
        ? "bitcast<f32>(b[index])"
        : dtype === "F16"
          ? this.nativeF16
            ? "f32(b[index])"
            : "unpack2x16float(b[index/2u])[index%2u]"
          : "f32((i32((b[index/4u]>>((index%4u)*8u))&255u)<<24)>>24)*d[row]";
    return (
      (dtype === "F16" && this.nativeF16 ? "enable f16;\n" : "") +
      params +
      `
 @group(0) @binding(1) var<storage,read> a:array<f32>;@group(0) @binding(2) var<storage,read> b:array<${dtype === "F16" && this.nativeF16 ? "f16" : "u32"}>;@group(0) @binding(3) var<storage,read> c:array<f32>;@group(0) @binding(4) var<storage,read> d:array<f32>;@group(0) @binding(5) var<storage,read_write> o:array<f32>;
 fn w(row:u32,k:u32)->f32{let index=row*u(2)+k;return ${read};}
 var<workgroup> aa:array<array<f32,16>,8>;var<workgroup> bb:array<array<f32,16>,8>;
 @compute @workgroup_size(8,8) fn main(@builtin(local_invocation_id) local:vec3<u32>,@builtin(workgroup_id) group:vec3<u32>){
 let M=u(1);let K=u(2);let N=u(3);let row=group.y*8u+local.y;let col=group.x*8u+local.x;var sums:array<f32,16>;var sum=0.0;
 for(var base=0u;base<K;base+=16u){for(var j=0u;j<2u;j++){let ak=base+local.x+j*8u;var av=0.0;if(row<M&&ak<K){av=a[row*K+ak];}aa[local.y][local.x+j*8u]=av;let bk=base+local.y+j*8u;var bv=0.0;if(col<N&&bk<K){bv=w(col,bk);}bb[local.x][local.y+j*8u]=bv;}workgroupBarrier();for(var k=0u;k<16u;k++){sums[k]+=aa[local.y][k]*bb[local.x][k];}workgroupBarrier();}
 if(row<M&&col<N){sum=(((sums[0]+sums[1])+(sums[2]+sums[3]))+((sums[4]+sums[5])+(sums[6]+sums[7])))+(((sums[8]+sums[9])+(sums[10]+sums[11]))+((sums[12]+sums[13])+(sums[14]+sums[15])))+c[col];if(u(4)>0u){sum=max(sum,0.0);}o[row*N+col]=sum;}}`
    );
  }
  async finish(outputs) {
    let staging = [];
    try {
      for (const t of outputs) {
        const b = this.buffer(
          product(t.shape) * 4,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        );
        staging.push(b);
        this.encoder.copyBufferToBuffer(t.buffer, 0, b, 0, b.size);
      }
      this.device.queue.writeBuffer(
        this.parameterBuffer,
        0,
        this.parameterData,
        0,
        this.dispatches * 64,
      );
      this.device.queue.submit([this.encoder.finish()]);
      this.encoder = null;
      await Promise.all(staging.map((b) => b.mapAsync(GPUMapMode.READ)));
      return staging.map((b) => new Float32Array(b.getMappedRange().slice(0)));
    } finally {
      for (const b of staging) {
        b.unmap();
        b.destroy();
      }
      this.encoder = null;
      this.collect();
    }
  }
  cancel() {
    this.encoder = null;
    this.collect();
  }
  memoryUsage() {
    return {
      bytes: this.weightBytes + this.scratchBytes,
      weightBytes: this.weightBytes,
      scratchBytes: this.scratchBytes,
      transientBuffers: this.live.size,
      peakBytes: this.peakBytes,
    };
  }
  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    this.cancel();
    for (const t of this.pool) t.buffer.destroy();
    for (const t of Object.values(this.weights)) {
      t.buffer.destroy();
      t.scales?.destroy();
    }
    this.parameterBuffer.destroy();
    this.pool = [];
    this.weights = {};
    this.weightBytes = 0;
    this.scratchBytes = 0;
    this.pipelines.clear();
  }
}
