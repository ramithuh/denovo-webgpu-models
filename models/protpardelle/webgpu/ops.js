// All activation arithmetic and learned operations execute as WGSL compute.
import { graphOps } from "./graph.js";
import { constants } from "../shared/constants.js";
const U = () => GPUBufferUsage;
const scalar = (x) => `${Number(x).toExponential(12)}`;
// Range-reduced polynomials avoid native trig error at large chain-gap angles.
const trig = `fn preciseSinCos(x:f32)->vec2<f32>{
let q=round(x*0.6366197723675814);var r=fma(-q,1.5703125,x);r=fma(-q,0.0004837512969970703125,r);r=fma(-q,0.00000007549789954891882,r);
let z=r*r;let sp=fma(z,fma(z,fma(z,fma(z,-0.00000002505210838544172,0.0000027557319223985893),-0.0001984126984126984),0.008333333333333333),-0.16666666666666666);let s=fma(r*z,sp,r);
let cp=fma(z,fma(z,fma(z,fma(z,-0.0000002755731922398589,0.0000248015873015873),-0.001388888888888889),0.041666666666666664),-0.5);let c=fma(z,cp,1.0);let quadrant=i32(q)&3;
if(quadrant==0){return vec2<f32>(s,c);}if(quadrant==1){return vec2<f32>(c,-s);}if(quadrant==2){return vec2<f32>(-s,-c);}return vec2<f32>(-c,s);}`;
export class GPUOps {
  constructor(device, weights) {
    this.device = device;
    this.weights = {};
    this.buffers = new Map();
    this.pipelines = new Map();
    this.pool = new Map();
    this.pinned = new Set();
    try {
      for (const [key, value] of Object.entries(weights)) {
        const buffer = this.allocate(value.data.byteLength, true);
        device.queue.writeBuffer(
          buffer,
          0,
          new Uint8Array(
            value.data.buffer,
            value.data.byteOffset,
            value.data.byteLength,
          ),
        );
        let scale;
        if (value.scale) {
          scale = this.allocate(value.scale.byteLength, true);
          device.queue.writeBuffer(scale, 0, value.scale);
        }
        this.weights[key] = { ...value, data: undefined, buffer, scale };
      }
      this.zero = this.allocate(4, true);
    } catch (error) {
      this.destroy();
      throw error;
    }
  }
  allocate(bytes, resident = false) {
    const size = Math.max(4, Math.ceil(bytes / 4) * 4);
    if (
      size > this.device.limits.maxBufferSize ||
      size > this.device.limits.maxStorageBufferBindingSize
    )
      throw new Error(
        `WebGPU buffer admission failed: ${size} bytes exceeds device storage limit`,
      );
    if (!resident && this.pool.get(size)?.length) {
      const buffer = this.pool.get(size).pop();
      this.buffers.get(buffer).pooled = false;
      return buffer;
    }
    const buffer = this.device.createBuffer({
      size,
      usage: U().STORAGE | U().COPY_DST | U().COPY_SRC,
    });
    this.buffers.set(buffer, { size, resident, pooled: false });
    return buffer;
  }
  tensor(rows, cols, data) {
    const buffer = this.allocate(rows * cols * 4);
    if (data)
      this.device.queue.writeBuffer(
        buffer,
        0,
        data instanceof Float32Array ? data : Float32Array.from(data),
      );
    return { buffer, rows, cols };
  }
  start() {
    this.encoder = this.device.createCommandEncoder();
    this.inputs = new Map();
  }
  attach(weights) {
    for (const [key, value] of Object.entries(weights)) {
      if (!key.startsWith("mpnn_model.") || this.weights[key]) continue;
      const buffer = this.allocate(value.data.byteLength, true);
      this.device.queue.writeBuffer(
        buffer,
        0,
        new Uint8Array(
          value.data.buffer,
          value.data.byteOffset,
          value.data.byteLength,
        ),
      );
      let scale;
      if (value.scale) {
        scale = this.allocate(value.scale.byteLength, true);
        this.device.queue.writeBuffer(scale, 0, value.scale);
      }
      this.weights[key] = { ...value, data: undefined, buffer, scale };
    }
  }
  async stepBoundary(keep) {
    this.device.queue.submit([this.encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    const retained = new Set(keep.filter(Boolean).map((t) => t.buffer));
    for (const buffer of this.traceBuffers ?? []) retained.add(buffer);
    for (const [buffer, info] of this.buffers)
      if (!info.resident && !retained.has(buffer)) {
        buffer.destroy();
        this.buffers.delete(buffer);
      }
    this.pool.clear();
    this.inputs.clear();
    this.pinned = new Set(this.traceBuffers ?? []);
    this.encoder = this.device.createCommandEncoder();
  }
  input(data, length) {
    if (data?.buffer instanceof GPUBuffer) return data.buffer;
    const key = data ?? `zero-${length}`;
    if (!this.inputs.has(key)) {
      const value = this.tensor(1, length, data ?? new Float32Array(length));
      this.inputs.set(key, value);
      this.pinned.add(value.buffer);
    }
    return this.inputs.get(key).buffer;
  }
  dispatch(source, bindings, groups, workgroup = "64") {
    const code = source.includes("@compute")
      ? source
      : `${trig}\n${bindings.map((b, i) => `@group(0) @binding(${i}) var<storage, ${i === bindings.length - 1 ? "read_write" : "read"}> b${i}: array<f32>;`).join("\n")}\n@compute @workgroup_size(${workgroup}) fn main(@builtin(global_invocation_id) id: vec3<u32>) { ${source} }`;
    let pipeline = this.pipelines.get(code);
    if (!pipeline) {
      const bindLayout = this.device.createBindGroupLayout({
        entries: bindings.map((_, binding) => ({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: new RegExp(
              "@binding\\(\\s*" +
                binding +
                "\\s*\\)[^;]*var<\\s*storage\\s*,\\s*read_write\\s*>",
            ).test(code)
              ? "storage"
              : "read-only-storage",
          },
        })),
      });
      pipeline = this.device.createComputePipeline({
        layout: this.device.createPipelineLayout({
          bindGroupLayouts: [bindLayout],
        }),
        compute: {
          module: this.device.createShaderModule({ code }),
          entryPoint: "main",
        },
      });
      this.pipelines.set(code, pipeline);
    }
    const group = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindings.map((buffer, binding) => ({
        binding,
        resource: { buffer },
      })),
    });
    const pass = this.encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(...groups);
    pass.end();
  }
  element(rows, cols, inputs, expr) {
    const out = this.tensor(rows, cols),
      count = rows * cols;
    this.dispatch(
      `let i=id.x; if(i>=${count}u){return;} let r=i/${cols}u; let c=i%${cols}u; b${inputs.length}[i]=${expr};`,
      [...inputs.map((x) => x.buffer ?? x), out.buffer],
      [Math.ceil(count / 64)],
    );
    return out;
  }
  weight(name) {
    const w = this.weights[name];
    if (!w) throw new Error(`Missing weight: ${name}`);
    return w;
  }
  linear(x, prefix) {
    const w = this.weight(prefix + ".weight"),
      bias = this.weights[prefix + ".bias"],
      [n, k] = w.shape,
      m = x.rows;
    if (x.cols !== k) throw new Error(`Linear shape mismatch ${prefix}`);
    const out = this.tensor(m, n);
    const weightType = w.dtype === "F32" ? "f32" : "u32";
    const load =
      w.dtype === "F32"
        ? "return W[i];"
        : w.dtype === "F16"
          ? "let pair=unpack2x16float(W[i/2u]); return pair[i%2u];"
          : `let bits=(W[i/4u]>>((i%4u)*8u))&255u; let value=i32(bits<<24u)>>24; return f32(value)*S[i/${k}u];`;
    const source = `
@group(0) @binding(0) var<storage,read> X:array<f32>;
@group(0) @binding(1) var<storage,read> W:array<${weightType}>;
@group(0) @binding(2) var<storage,read> B:array<f32>;
@group(0) @binding(3) var<storage,read> S:array<f32>;
@group(0) @binding(4) var<storage,read_write> Y:array<f32>;
var<workgroup> A:array<f32,256>; var<workgroup> T:array<f32,256>;
fn weight(i:u32)->f32 { ${load} }
@compute @workgroup_size(16,16) fn main(@builtin(local_invocation_id) l:vec3<u32>,@builtin(workgroup_id) g:vec3<u32>){
let row=g.y*16u+l.y; let col=g.x*16u+l.x; var sum=${bias ? `B[min(col,${n - 1}u)]` : "B[0]"};
for(var base=0u;base<${k}u;base+=16u){
var a=0.0; var w=0.0;
if(row<${m}u && base+l.x<${k}u){a=X[row*${k}u+base+l.x];}
if(col<${n}u && base+l.y<${k}u){w=weight(col*${k}u+base+l.y);}
A[l.y*16u+l.x]=a; T[l.y*16u+l.x]=w; workgroupBarrier();
for(var j=0u;j<16u;j++){sum=fma(A[l.y*16u+j],T[j*16u+l.x],sum);} workgroupBarrier();
}
if(row<${m}u && col<${n}u){Y[row*${n}u+col]=sum;}
// Retain the scale binding even for nonquantized storage.
if(row==4294967295u){Y[0]=S[0];}
}`;
    this.dispatch(
      source,
      [
        x.buffer,
        w.buffer,
        bias?.buffer ?? this.zero,
        w.scale ?? this.zero,
        out.buffer,
      ],
      [Math.ceil(n / 16), Math.ceil(m / 16)],
    );
    return out;
  }
  norm(x, prefix) {
    const w =
        this.weights[prefix + ".gamma"] ?? this.weight(prefix + ".weight"),
      b = this.weights[prefix + ".beta"] ?? this.weight(prefix + ".bias");
    const out = this.tensor(x.rows, x.cols),
      d = x.cols;
    this.dispatch(
      `let r=id.x; if(r>=${x.rows}u){return;} var mean=0.0;var correction=0.0; for(var j=0u;j<${d}u;j++){let y=b0[r*${d}u+j]-correction;let next=mean+y;correction=(next-mean)-y;mean=next;} mean/=f32(${d}); var variance=0.0;correction=0.0; for(var j=0u;j<${d}u;j++){let v=b0[r*${d}u+j]-mean;let y=v*v-correction;let next=variance+y;correction=(next-variance)-y;variance=next;} let inv=inverseSqrt(variance/f32(${d})+0.00001); for(var j=0u;j<${d}u;j++){b3[r*${d}u+j]=(b0[r*${d}u+j]-mean)*inv*b1[j]+b2[j];}`,
      [x.buffer, w.buffer, b.buffer, out.buffer],
      [Math.ceil(x.rows / 64)],
    );
    return out;
  }
  silu(x) {
    return this.element(x.rows, x.cols, [x], "b0[i]/(1.0+exp(-b0[i]))");
  }
  add(x, y) {
    return this.element(x.rows, x.cols, [x, y], "b0[i]+b1[i]");
  }
  mask(x, data) {
    return this.element(
      x.rows,
      x.cols,
      [x, this.input(data, x.rows)],
      "b0[i]*b1[r]",
    );
  }
  modulate(x, t, mode) {
    const offset = `(r%${t.rows}u)*${t.cols}u+c`;
    return this.element(
      x.rows,
      x.cols,
      [x, t],
      mode === "gate"
        ? `b0[i]*(b1[${offset}+${2 * x.cols}u]+1.0)`
        : `b0[i]*(b1[${offset}]+1.0)+b1[${offset}+${x.cols}u]`,
    );
  }
  noise(input, d) {
    const out = this.tensor(input.seqMask.length, d),
      freq = constants.noiseFrequencies[d].map(scalar).join(",");
    this.dispatch(
      `let i=id.x;if(i>=${input.seqMask.length * d}u){return;}let r=i/${d}u;let c=i%${d}u;let frequencies=array<f32,${d / 2}>(${freq});let cs=preciseSinCos(0.25*log(max(b0[r],0.000001))*frequencies[c%${d / 2}u]);b1[i]=select(cs.y,cs.x,c>=${d / 2}u);`,
      [this.input(input.noiseLevel, input.seqMask.length), out.buffer],
      [Math.ceil((input.seqMask.length * d) / 64)],
    );
    return out;
  }
  features(input, config) {
    const m = config.model,
      a = m.struct_model.n_atoms,
      n = input.seqMask.length,
      sd = config.data.sigma_data;
    const concat =
        m.crop_conditional && m.conditioning_style.includes("concat"),
      hot = m.crop_conditional && m.conditioning_style.includes("hotspot");
    const channels = 6 + (concat ? 3 : 0) + (hot ? 1 : 0),
      out = this.tensor(n, a * channels);
    this.dispatch(
      `let i=id.x; if(i>=${n * a * channels}u){return;} let r=i/${a * channels}u;let c=(i%${a * channels}u)/${a}u;let atom=i%${a}u;let j=r*${a * 3}u+atom*3u;var v=0.0;
if(c<3u){let s=max(0.000001,b1[r]);v=b0[j+c]/sqrt(s*s+${scalar(sd * sd)});}else if(c<6u){v=b2[j+c-3u];}else if(c<9u){v=b3[j+c-6u]/${scalar(sd)};}else if(atom==0u || atom==1u || atom==2u || atom==4u){v=b4[r];} b5[i]=v;`,
      [
        this.input(input.coords, n * a * 3),
        this.input(input.noiseLevel, n),
        this.input(input.selfCond, n * a * 3),
        this.input(input.cropCond, n * a * 3),
        this.input(input.hotspotMask, n),
        out.buffer,
      ],
      [Math.ceil((n * a * channels) / 64)],
    );
    return out;
  }
  attention(q, kv, input, config, prefix) {
    const n = q.rows,
      d = q.cols,
      {
        n_heads: heads,
        dim_head: dh,
        position_embedding_type: position,
        position_embedding_max: max,
      } = config.model.struct_model.uvit;
    const rotary = position.includes("rotary"),
      relpos = position.includes("relative"),
      chain = position.includes("relchain");
    const queries = this.tensor(n, d),
      keys = this.tensor(n, d);
    const freq = rotary ? this.weight(prefix + "rope.freqs").buffer : this.zero;
    for (const [isQuery, out] of [
      [true, queries],
      [false, keys],
    ]) {
      const at = isQuery ? "i" : `r*${d * 2}u+c`,
        other = isQuery ? "r*" + d + "u+(c^1u)" : `r*${d * 2}u+(c^1u)`;
      this.dispatch(
        `let i=id.x;if(i>=${n * d}u){return;}let r=i/${d}u;let c=i%${d}u;let scale=${isQuery ? scalar(dh ** -0.5) : "1.0"}; let x=b0[${at}]*scale;
${rotary ? `let angle=b1[r]*b3[(c%${dh}u)/2u]+${input.chainIndex ? "b2[r]*b3[0]" : "0.0*b2[r]"};let other=b0[${other}]*scale;let cs=preciseSinCos(angle); b4[i]=x*cs.y+select(-other,other,c%2u==1u)*cs.x;` : "b4[i]=x+0.0*(b1[r]+b2[r]+b3[0]);"}`,
        [
          isQuery ? q.buffer : kv.buffer,
          this.input(input.residueIndex, n),
          this.input(input.chainIndex, n),
          freq,
          out.buffer,
        ],
        [Math.ceil((n * d) / 64)],
      );
    }
    const scores = this.tensor(heads * n, n),
      probs = this.tensor(heads * n, n),
      result = this.tensor(n, d);
    // Four small positional arrays share a single packed F32 buffer to remain below binding limits.
    if (!this.positional) this.positional = new Map();
    let positional = this.positional.get(prefix);
    if (!positional) {
      const size = heads * (max * 2 + 2),
        packed = new Float32Array(size * 2);
      positional = this.allocate(packed.byteLength, true);
      this.device.queue.writeBuffer(positional, 0, packed);
      for (const [kind, offset] of [
        ["relpos", 0],
        ["relchain", size * 4],
      ]) {
        const w = this.weights[prefix + kind + ".0.linear.weight"],
          b = this.weights[prefix + kind + ".0.linear.bias"];
        if (w) {
          this.encoder.copyBufferToBuffer(
            w.buffer,
            0,
            positional,
            offset,
            heads * (max * 2 + 1) * 4,
          );
          this.encoder.copyBufferToBuffer(
            b.buffer,
            0,
            positional,
            offset + heads * (max * 2 + 1) * 4,
            heads * 4,
          );
        }
      }
      this.positional.set(prefix, positional);
    }
    const bins = max * 2 + 1,
      span = heads * (bins + 1);
    this.dispatch(
      `let idx=id.x;if(idx>=${heads * n * n}u){return;}let j=idx%${n}u;let i=(idx/${n}u)%${n}u;let h=idx/${n * n}u;var sum=0.0;for(var c=0u;c<${dh}u;c++){sum+=b0[i*${d}u+h*${dh}u+c]*b1[j*${d}u+h*${dh}u+c];}var bias=0.0;
${chain ? `let ci=select(0u,1u,b4[i]!=b4[j])+${max}u;bias=b5[${span}u+h*${bins}u+ci]+b5[${span + heads * bins}u+h];` : ""}
${relpos ? `let ri=u32(clamp(b3[i]-b3[j],-${max}.0,${max}.0)+${max}.0);bias+=b5[h*${bins}u+ri]+b5[${heads * bins}u+h];${chain ? "bias*=2.0;" : ""}` : ""}
b6[idx]=sum+bias-(1.0-b2[i]*b2[j])*1000000.0+0.0*(b3[0]+b4[0]+b5[0]);`,
      [
        queries.buffer,
        keys.buffer,
        this.input(input.seqMask, n),
        this.input(input.residueIndex, n),
        this.input(input.chainIndex, n),
        positional,
        scores.buffer,
      ],
      [Math.ceil((heads * n * n) / 64)],
    );
    this.dispatch(
      `let r=id.x;if(r>=${heads * n}u){return;}var maxval=-3.402823e38;for(var j=0u;j<${n}u;j++){maxval=max(maxval,b0[r*${n}u+j]);}var total=0.0;for(var j=0u;j<${n}u;j++){total+=exp(b0[r*${n}u+j]-maxval);}for(var j=0u;j<${n}u;j++){b1[r*${n}u+j]=exp(b0[r*${n}u+j]-maxval)/total;}`,
      [scores.buffer, probs.buffer],
      [Math.ceil((heads * n) / 64)],
    );
    this.dispatch(
      `let idx=id.x;if(idx>=${n * d}u){return;}let i=idx/${d}u;let h=(idx%${d}u)/${dh}u;let c=idx%${dh}u;var value=0.0;for(var j=0u;j<${n}u;j++){value+=b0[(h*${n}u+i)*${n}u+j]*b1[j*${d * 2}u+${d}u+h*${dh}u+c];}b2[idx]=value;`,
      [probs.buffer, kv.buffer, result.buffer],
      [Math.ceil((n * d) / 64)],
    );
    return {
      ...result,
      attentionCache: { queries, keys, probs, kv, input, config, prefix },
    };
  }
  output(patch, input, config) {
    const a = config.model.struct_model.n_atoms,
      sd = config.data.sigma_data,
      n = patch.rows;
    return this.element(
      n,
      a * 3,
      [
        patch,
        this.input(input.coords, n * a * 3),
        this.input(input.noiseLevel, n),
        this.input(input.seqMask, n),
      ],
      `(b0[r*${a * 3}u+(c%3u)*${a}u+c/3u]*(max(b2[r],0.000001)*${scalar(sd)}/sqrt(max(b2[r],0.000001)*max(b2[r],0.000001)+${scalar(sd * sd)}))+b1[i]*(${scalar(sd * sd)}/(max(b2[r],0.000001)*max(b2[r],0.000001)+${scalar(sd * sd)})))*b3[r]`,
    );
  }
  beginBlock() {
    this.blockBuffers = new Set(this.buffers.keys());
  }
  endBlock(keep) {
    const retained = new Set(keep.map((t) => t.buffer));
    for (const [buffer, info] of this.buffers)
      if (
        !info.resident &&
        !info.pooled &&
        !this.blockBuffers.has(buffer) &&
        !retained.has(buffer) &&
        !this.pinned.has(buffer)
      ) {
        info.pooled = true;
        if (!this.pool.has(info.size)) this.pool.set(info.size, []);
        this.pool.get(info.size).push(buffer);
      }
  }
  async readback(tensors) {
    const records = [],
      staging = [];
    try {
      for (const [key, tensor] of Object.entries(tensors)) {
        const size = tensor.rows * tensor.cols * 4,
          buffer = this.device.createBuffer({
            size,
            usage: U().MAP_READ | U().COPY_DST,
          });
        staging.push(buffer);
        this.buffers.set(buffer, { size, resident: false });
        this.encoder.copyBufferToBuffer(tensor.buffer, 0, buffer, 0, size);
        records.push([key, buffer]);
      }
      this.device.queue.submit([this.encoder.finish()]);
      const result = {};
      for (const [key, buffer] of records) {
        await buffer.mapAsync(GPUMapMode.READ);
        result[key] = new Float32Array(buffer.getMappedRange().slice(0));
        buffer.unmap();
      }
      return result;
    } finally {
      for (const buffer of staging) {
        buffer.destroy();
        this.buffers.delete(buffer);
      }
    }
  }
  cleanup() {
    for (const [buffer, info] of this.buffers)
      if (!info.resident) {
        buffer.destroy();
        this.buffers.delete(buffer);
      }
    this.pool.clear();
    this.pinned.clear();
    this.inputs?.clear();
  }
  memoryUsage() {
    let bytes = 0,
      buffers = 0,
      transientBuffers = 0,
      transientBytes = 0;
    for (const info of this.buffers.values()) {
      buffers++;
      bytes += info.size;
      if (!info.resident) {
        transientBuffers++;
        transientBytes += info.size;
      }
    }
    return { buffers, bytes, transientBuffers, transientBytes };
  }
  destroy() {
    for (const buffer of this.buffers.keys()) buffer.destroy();
    this.buffers.clear();
    this.pipelines.clear();
    this.pool.clear();
    this.pinned.clear();
  }
}
Object.assign(GPUOps.prototype, graphOps);
