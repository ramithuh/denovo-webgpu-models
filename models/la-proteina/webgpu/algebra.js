import { edgeSoftmax } from "./edge-softmax.js";
import { numericWGSL } from "../shared/numeric.js";
import { Algebra, sizeOf } from "../shared/tensor.js";
export class GPUAlgebra extends Algebra {
  constructor(device, {useF16=false}={}) {
    super();
    this.device = device;
    this.useF16=useF16&&device.features.has('shader-f16');
    this.buffers = new Set();
    this.persistent = new Set();
    this.pipelines = new Map();
    this.encoder = null;
    this.dispatches = 0;
    this.pool = new Map();
    this.available = new Set();
    this.validationError = null;
    this.destroyed=false;
    this.allocations=0;
    this.reusedBuffers=0;
    this.listener = (e) => {
      this.validationError = e.error;
    };
    device.addEventListener("uncapturederror", this.listener);
  }
  alloc(shape, persistent = false) {
    if(this.destroyed)throw Error('GPU algebra destroyed');
    const size = sizeOf(shape),
      bytes = Math.max(4, size * 4),
      limit = Math.min(
        this.device.limits.maxBufferSize,
        this.device.limits.maxStorageBufferBindingSize,
      );
    if (bytes > limit) throw Error(`GPU buffer ${bytes} exceeds ${limit}`);
    const pooled=!persistent && this.pool.get(bytes)?.pop();
    if(pooled)this.reusedBuffers++;else this.allocations++;
    const buffer = pooled || this.device.createBuffer({
        size: bytes,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      });
    this.available.delete(buffer);
    this.buffers.add(buffer);
    if (persistent) this.persistent.add(buffer);
    return { shape, size, buffer };
  }
  upload(shape, data, persistent = false) {
    const out = this.alloc(shape, persistent);
    this.device.queue.writeBuffer(out.buffer, 0, data);
    return out;
  }
  uploadWeight(t) {
    if(this.destroyed)throw Error('GPU algebra destroyed');
    if (t.dtype === "F32") return this.upload(t.shape, t.data, true);
    const rawBytes = t.data.byteLength,
      scaleOffset = Math.ceil(rawBytes / 4),
      bytes = (scaleOffset + (t.scale?.length ?? 0)) * 4,
      limit = Math.min(
        this.device.limits.maxBufferSize,
        this.device.limits.maxStorageBufferBindingSize,
      );
    if (bytes > limit) throw Error("Weight exceeds GPU limits");
    const packed = new Uint8Array(bytes);
    packed.set(
      new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength),
    );
    if (t.scale) new Float32Array(packed.buffer, scaleOffset * 4).set(t.scale);
    const buffer = this.device.createBuffer({
      size: bytes,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    this.allocations++;
    this.buffers.add(buffer);
    this.persistent.add(buffer);
    this.device.queue.writeBuffer(buffer, 0, packed);
    return {
      shape: t.shape,
      size: sizeOf(t.shape),
      buffer,
      dtype: t.dtype,
      scaleOffset,
    };
  }
  linear(x,w,b) {
    const K=x.shape.at(-1),N=w.shape[0],M=x.size/K;
    const nativeF16=w.dtype==='F16'&&this.useF16;
    let load='W[index]';
    if(w.dtype==='F16')load='unpack2x16float(W[index/2u])[index%2u]';
    if(nativeF16)load='f32(W[index])';
    if(w.dtype==='I8')load=`f32(i32((W[index/4u]>>((index%4u)*8u))<<24u)>>24)*bitcast<f32>(W[${w.scaleOffset}u+index/${K}u])`;
    const code=`${nativeF16?'enable f16;':''}
@group(0) @binding(0) var<storage,read> X:array<f32>;
@group(0) @binding(1) var<storage,read> W:array<${nativeF16?'f16':w.dtype&&w.dtype!=='F32'?'u32':'f32'}>;
${b?'@group(0) @binding(2) var<storage,read> B:array<f32>;':''}
@group(0) @binding(${b?3:2}) var<storage,read_write> Y:array<f32>;
fn weight(index:u32)->f32{return ${load};}
var<workgroup> xs:array<f32,256>;
var<workgroup> ws:array<f32,256>;
@compute @workgroup_size(16,16) fn main(@builtin(local_invocation_id) l:vec3<u32>,@builtin(workgroup_id) g:vec3<u32>){
 let row=g.y*16u+l.y;let col=g.x*16u+l.x;
 // PyTorch's addmm initializes the accumulator from the bias before GEMM.
 var sum=0.0;${b?'if(col<'+N+'u){sum=B[col];}':''}
 for(var base=0u;base<${K}u;base+=16u){
  var a=0.0;var v=0.0;
  if(row<${M}u && base+l.x<${K}u){a=X[row*${K}u+base+l.x];}
  if(col<${N}u && base+l.y<${K}u){v=weight(col*${K}u+base+l.y);}
  xs[l.y*16u+l.x]=a;ws[l.y*16u+l.x]=v;
  workgroupBarrier();
  for(var k=0u;k<16u;k++){sum=fma(xs[l.y*16u+k],ws[k*16u+l.x],sum);}
  workgroupBarrier();
 }
 if(row<${M}u && col<${N}u){Y[row*${N}u+col]=sum;}
}`;
    let pipeline=this.pipelines.get(code);
    if(!pipeline){pipeline=this.device.createComputePipeline({layout:'auto',compute:{module:this.device.createShaderModule({code}),entryPoint:'main'}});this.pipelines.set(code,pipeline);}
    const out=this.alloc([...x.shape.slice(0,-1),N]);
    const inputs=b?[x,w,b,out]:[x,w,out];
    const bind=this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:inputs.map((t,binding)=>({binding,resource:{buffer:t.buffer}}))});
    this.encoder??=this.device.createCommandEncoder();const pass=this.encoder.beginComputePass();
    pass.setPipeline(pipeline);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(Math.ceil(N/16),Math.ceil(M/16));pass.end();
    if(++this.dispatches%128===0)this.flush();return out;
  }
  kernel(shape, inputs, js, wgsl) {
    const size = sizeOf(shape);
    const rewrite = (code) => {
      let result = "",
        last = 0;
      const re = /a(\d+)\[/g;
      let match;
      while ((match = re.exec(code))) {
        let depth = 1,
          end = re.lastIndex;
        while (depth && end < code.length) {
          if (code[end] === "[") depth++;
          if (code[end] === "]") depth--;
          end++;
        }
        result +=
          code.slice(last, match.index) +
          `load${match[1]}(${rewrite(code.slice(re.lastIndex, end - 1))})`;
        last = end;
        re.lastIndex = end;
      }
      return result + code.slice(last);
    };
    const declarations = inputs
      .map((t, j) => {
        const raw = t.dtype && t.dtype !== "F32",
          binding = `@group(0) @binding(${j}) var<storage,read> input${j}:array<${raw ? "u32" : "f32"}>;`;
        let expression = `input${j}[index]`;
        if (t.dtype === "F16")
          expression = `unpack2x16float(input${j}[index/2u])[index%2u]`;
        if (t.dtype === "I8")
          expression = `f32(i32((input${j}[index/4u]>>((index%4u)*8u))<<24u)>>24)*bitcast<f32>(input${j}[${t.scaleOffset}u+index/${t.shape[1]}u])`;
        return binding + `fn load${j}(index:u32)->f32{return ${expression};}`;
      })
      .join("\n");
    const accurateMath = `fn rfdiv(a:f32,b:f32)->f32 {
 let signBit=(bitcast<u32>(a)^bitcast<u32>(b))&0x80000000u;
 let x=abs(a);let y=abs(b);var bits=bitcast<u32>(x/y);
 if(bits==0u || bits>=0x7f800000u){return a/b;}
 for(var k=0u;k<4u;k++){
  let q=bitcast<f32>(bits);let e=fma(-q,y,x);
  let up=e>0.0;let nextBits=select(bits-1u,bits+1u,up);
  let halfGap=abs(bitcast<f32>(nextBits)-q)*0.5;
  let boundary=halfGap*y;
  if(abs(e)>boundary || (abs(e)==boundary && (bits&1u)==1u)){bits=nextBits;}else{break;}
 }
 return bitcast<f32>(bits|signBit);
}
fn rfSqrt(x:f32)->f32 {let y=sqrt(x);if(x==0.0){return 0.0;}return y+fma(-y,y,x)/(2.0*y);}
`;
    const code =
      numericWGSL +
      accurateMath +
      declarations +
      `@group(0) @binding(${inputs.length}) var<storage,read_write> o:array<f32>;\n@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){let i=gid.x+gid.y*65535u*64u;if(i>=${size}u){return;}${rewrite(wgsl).replaceAll("sqrt(", "rfSqrt(")}}`;
    let pipeline = this.pipelines.get(code);
    if (!pipeline) {
      const module = this.device.createShaderModule({ code });
      pipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
      this.pipelines.set(code, pipeline);
    }
    const out = this.alloc(shape),
      bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [...inputs, out].map((t, binding) => ({
          binding,
          resource: { buffer: t.buffer },
        })),
      });
    this.encoder ??= this.device.createCommandEncoder();
    const pass = this.encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    const groups = Math.ceil(size / 64);
    pass.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535));
    pass.end();
    if (++this.dispatches % 128 === 0) this.flush();
    return out;
  }
  softmax(x, axis = -1, dgl = false) {
    return dgl ? edgeSoftmax(this, x, axis) : super.softmax(x, axis, false);
  }
  groupInverse(mom, eps) {
    const shape = [mom.shape[0]],
      epsLo = eps - Math.fround(eps),
      x = this.kernel(shape, [mom], "", `o[i]=a0[i*2u+1u]+${eps};`),
      r = this.kernel(shape, [x], "", "o[i]=rfdiv(1.0,sqrt(a0[i]));"),
      square = this.mul(r, r);
    return this.kernel(
      shape,
      [mom, x, r, square],
      "",
      `let v=a0[i*2u+1u];let x=a1[i];let r=a2[i];let r2=a3[i];let low=((v-x)+${eps})+${epsLo};let r2low=fma(r,r,-r2);let error=(fma(-r2,x,1.0)-r2low*x)-r2*low;o[i]=r+(0.5*r)*error;`,
    );
  }
  flush() {
    if (this.encoder) {
      this.device.queue.submit([this.encoder.finish()]);
      this.encoder = null;
    }
  }
  async read(t) {
    if(this.destroyed)throw Error('GPU algebra destroyed');
    this.flush();
    const b = this.device.createBuffer({
      size: Math.max(4, t.size * 4),
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.buffers.add(b);
    try {
      const e = this.device.createCommandEncoder();
      e.copyBufferToBuffer(t.buffer, 0, b, 0, t.size * 4);
      this.device.queue.submit([e.finish()]);
      await b.mapAsync(GPUMapMode.READ);
      return new Float32Array(b.getMappedRange()).slice(0, t.size);
    } finally {
      b.destroy();
      this.buffers.delete(b);
    }
  }
  async checkpoint(keep = []) {
    this.flush();
    const live = new Set(
      keep
        .flat(Infinity)
        .filter(Boolean)
        .map((t) => t.buffer),
    );
    for (const b of this.buffers)
      if (!this.persistent.has(b) && !live.has(b)) {
        if (!this.available.has(b)) {
          const list = this.pool.get(b.size) ?? [];
          list.push(b);
          this.pool.set(b.size, list);
          this.available.add(b);
        }
      }
    await this.device.queue.onSubmittedWorkDone();
    if (this.validationError) throw this.validationError;
  }
  clear() {
    this.flush();
    this.pool.clear();
    this.available.clear();
    for (const b of this.buffers)
      if (!this.persistent.has(b)) {
        b.destroy();
        this.buffers.delete(b);
      }
  }
  destroy() {
    if(this.destroyed)return;
    this.destroyed=true;
    this.device.removeEventListener("uncapturederror", this.listener);
    this.pool.clear();
    this.available.clear();
    this.encoder = null;
    for (const b of this.buffers) b.destroy();
    this.buffers.clear();
    this.persistent.clear();
    this.pipelines.clear();
  }
  memoryUsage() {
    let bytes = 0,
      transientBuffers = 0;
    for (const b of this.buffers) {
      bytes += b.size;
      if (!this.persistent.has(b)) transientBuffers++;
    }
    return { bytes, transientBuffers, allocations:this.allocations,reusedBuffers:this.reusedBuffers };
  }
}
