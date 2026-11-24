import { Ops } from "../shared/ops.js";
const product = (s) => s.reduce((a, b) => a * b, 1);
const MAX_POOL_BYTES = 128 * 1024 * 1024;
const params = `struct Params{v:array<vec4<f32>,16>};@group(0) @binding(0) var<uniform> params:Params;fn pf(i:u32)->f32{return params.v[i/4u][i%4u];}fn u(i:u32)->u32{return u32(pf(i));}`;
const quat = `fn qmat(q:vec4<f32>)->mat3x3<f32>{let w=q.x;let x=q.y;let y=q.z;let z=q.w;return mat3x3<f32>(vec3<f32>(1.0-2.0*(y*y+z*z),2.0*(x*y+w*z),2.0*(x*z-w*y)),vec3<f32>(2.0*(x*y-w*z),1.0-2.0*(x*x+z*z),2.0*(y*z+w*x)),vec3<f32>(2.0*(x*z+w*y),2.0*(y*z-w*x),1.0-2.0*(x*x+y*y)));}`;
const matquat = `fn matquat(i:u32)->vec4<f32>{let r0=a[i];let r1=a[i+1u];let r2=a[i+2u];let r3=a[i+3u];let r4=a[i+4u];let r5=a[i+5u];let r6=a[i+6u];let r7=a[i+7u];let r8=a[i+8u];let d=vec4<f32>(1.0+r0+r4+r8,1.0+r0-r4-r8,1.0-r0+r4-r8,1.0-r0-r4+r8);var k=0u;for(var j=1u;j<4u;j++){if(d[j]>d[k]){k=j;}}let v=2.0*sqrt(max(0.0,d[k]));var q=vec4<f32>(0.0);if(k==0u){q=vec4<f32>(v/4.0,(r7-r5)/v,(r2-r6)/v,(r3-r1)/v);}else if(k==1u){q=vec4<f32>((r7-r5)/v,v/4.0,(r1+r3)/v,(r2+r6)/v);}else if(k==2u){q=vec4<f32>((r2-r6)/v,(r1+r3)/v,v/4.0,(r5+r7)/v);}else{q=vec4<f32>((r3-r1)/v,(r2+r6)/v,(r5+r7)/v,v/4.0);}return normalize(q);}`;

function fold2(groups) {
  if (groups <= 65535) return [Math.max(1, groups), 1];
  return [65535, Math.ceil(groups / 65535)];
}

export class GPUOps extends Ops {
  constructor(device, { useF16 = false } = {}) {
    super({});
    this.nativeF16 = useF16 && device.features.has("shader-f16");
    this.device = device;
    this.weights = {};
    this.live = new Set();
    this.pinned = new Set();
    this.pool = [];
    this.pendingDestroy = [];
    this.pipelines = new Map();
    this.hasUnvalidatedPipelines = true;
    this.weightBytes = 0;
    this.scratchBytes = 0;
    this.peakBytes = 0;
    this.destroyed = false;
    this.encoder = null;
    this.pass = null;
    this.dispatches = 0;
    this.parameterData = new Float32Array(64 * 4096);
    this.parameterBuffer = this.buffer(
      this.parameterData.byteLength,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    );
    this.scratchBytes += this.parameterData.byteLength;
    this.lastTransInput = null;
    this.lastRotInput = null;
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
    const i = this.pool.findIndex((t) => t.buffer.size === bytes);
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
  pinnedTensor(data, shape) {
    const t = this.tensor(data, shape);
    this.pinned.add(t);
    return t;
  }
  collect(keep = []) {
    const set = new Set(keep);
    for (const t of this.pinned) set.add(t);
    for (const t of this.live) {
      if (!set.has(t)) {
        this.live.delete(t);
        this.pool.push(t);
      }
    }
    // Never call buffer.destroy() while this.encoder is recording commands!
    // Defer pool eviction to _trimPool() / _drainPendingDestroy() after submit/cancel.
    if (!this.encoder) {
      this._trimPool();
      this._drainPendingDestroy();
    }
  }
  _trimPool() {
    let poolBytes = 0;
    for (const t of this.pool) poolBytes += t.buffer.size;
    while (poolBytes > MAX_POOL_BYTES && this.pool.length > 0) {
      const evicted = this.pool.shift();
      poolBytes -= evicted.buffer.size;
      if (this.encoder) {
        this.pendingDestroy.push(evicted.buffer);
      } else {
        this.scratchBytes = Math.max(0, this.scratchBytes - evicted.buffer.size);
        evicted.buffer.destroy();
      }
    }
  }
  _drainPendingDestroy() {
    if (this.encoder || this.pendingDestroy.length === 0) return;
    while (this.pendingDestroy.length > 0) {
      const b = this.pendingDestroy.pop();
      this.scratchBytes = Math.max(0, this.scratchBytes - b.size);
      b.destroy();
    }
  }
  _endPass() {
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
  }
  begin() {
    if (this.destroyed) throw Error("Engine destroyed");
    this.encoder = this.device.createCommandEncoder();
    this.pass = null;
    this.dispatches = 0;
    this.lastTransInput = null;
    this.lastRotInput = null;
  }
  frames(rot, trans) {
    this.lastRotInput = rot;
    this.lastTransInput = trans;
    return super.frames(rot, trans);
  }
  run(name, inputs, shape, p, body) {
    const out = this.alloc(shape),
      n = product(shape);
    if (!this.encoder) throw Error("No active GPU operation");
    if (this.dispatches >= 4096) throw Error("Dispatch capacity exceeded");

    let key = name;
    let source;
    let dispatchXY;

    if (name === "linear") {
      const dtype = inputs[1].dtype;
      const M = p[0],
        N = p[2];
      const tileKind = M <= 160 ? "32x64" : "64x128";
      key = `linear_${tileKind}_${dtype}${dtype === "F16" && this.nativeF16 ? "_native" : ""}`;
      source =
        tileKind === "32x64"
          ? this.linear32x64Source(dtype)
          : this.linear64x128Source(dtype);
      const groups =
        tileKind === "32x64"
          ? Math.ceil(M / 32) * Math.ceil(N / 64)
          : Math.ceil(M / 64) * Math.ceil(N / 128);
      dispatchXY = fold2(groups);
    } else if (name === "norm") {
      key = "norm_row";
      source = this.normRowSource();
      const C = p[0],
        rows = n / C;
      dispatchXY = fold2(rows);
    } else if (name === "softmax") {
      key = "softmax_row";
      source = this.softmaxRowSource();
      const N = p[0],
        rows = n / N;
      dispatchXY = fold2(rows);
    } else if (name === "ipaLogits" && p[2] <= 128 && p[3] <= 8) {
      key = "ipaLogits_tiled";
      source = this.ipaLogitsTiledSource();
      const L = p[0],
        H = p[1];
      const tiles = Math.ceil(L / 8);
      dispatchXY = fold2(H * tiles * tiles);
    } else {
      const binding =
        inputs
          .map(
            (_, i) =>
              `@group(0) @binding(${i + 1}) var<storage,read> ${String.fromCharCode(97 + i)}:array<f32>;`,
          )
          .join("\n") +
        `\n@group(0) @binding(${inputs.length + 1}) var<storage,read_write> o:array<f32>;`;
      source =
        params +
        "\n" +
        binding +
        "\n" +
        quat +
        (name === "frames" ? matquat : "") +
        `\n@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>,@builtin(num_workgroups) groups:vec3<u32>){let idx=id.x+id.y*groups.x*64u;if(idx>=u(0)){return;}${body.replace(/\bf\(/g, "pf(")}}`;
      const groups = Math.ceil(n / 64),
        width = Math.min(
          groups,
          this.device.limits.maxComputeWorkgroupsPerDimension,
        );
      dispatchXY = [Math.max(1, width), Math.max(1, Math.ceil(groups / Math.max(1, width)))];
    }

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
      this.hasUnvalidatedPipelines = true;
    }

    const slot = this.dispatches++;
    this.parameterData.set([n, ...p], slot * 64);
    const buffers = inputs.map((t) => t.buffer);
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
    if (!this.pass) this.pass = this.encoder.beginComputePass();
    this.pass.setPipeline(pipeline);
    this.pass.setBindGroup(0, bind);
    this.pass.dispatchWorkgroups(dispatchXY[0], dispatchXY[1]);
    return out;
  }

  normRowSource() {
    return `${params}
@group(0) @binding(1) var<storage,read> a:array<f32>;
@group(0) @binding(2) var<storage,read> b:array<f32>;
@group(0) @binding(3) var<storage,read> c:array<f32>;
@group(0) @binding(4) var<storage,read_write> o:array<f32>;
var<workgroup> red:array<f32,64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li:u32,@builtin(workgroup_id) wid:vec3<u32>){
  let C=u(1);
  let rows=u(0)/C;
  let r=wid.x+wid.y*65535u;
  let live=r<rows;
  let row=min(r,rows-1u);
  let base=row*C;
  var s=0.0;
  for(var k=li;k<C;k+=64u){s+=a[base+k];}
  red[li]=s;workgroupBarrier();
  if(li<32u){red[li]+=red[li+32u];}workgroupBarrier();
  if(li<16u){red[li]+=red[li+16u];}workgroupBarrier();
  if(li<8u){red[li]+=red[li+8u];}workgroupBarrier();
  if(li==0u){red[0]=((red[0]+red[4])+(red[2]+red[6]))+((red[1]+red[5])+(red[3]+red[7]));}
  workgroupBarrier();
  let mean=red[0]/f32(C);
  workgroupBarrier();
  var q=0.0;
  for(var k=li;k<C;k+=64u){let d=a[base+k]-mean;q+=d*d;}
  red[li]=q;workgroupBarrier();
  if(li<32u){red[li]+=red[li+32u];}workgroupBarrier();
  if(li<16u){red[li]+=red[li+16u];}workgroupBarrier();
  if(li<8u){red[li]+=red[li+8u];}workgroupBarrier();
  if(li==0u){red[0]=((red[0]+red[4])+(red[2]+red[6]))+((red[1]+red[5])+(red[3]+red[7]));}
  workgroupBarrier();
  let inv=inverseSqrt(red[0]/f32(C)+0.00001);
  if(!live){return;}
  for(var k=li;k<C;k+=64u){
    o[base+k]=(a[base+k]-mean)*inv*b[k]+c[k];
  }
}`;
  }

  softmaxRowSource() {
    return `${params}
@group(0) @binding(1) var<storage,read> a:array<f32>;
@group(0) @binding(2) var<storage,read_write> o:array<f32>;
var<workgroup> red:array<f32,64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li:u32,@builtin(workgroup_id) wid:vec3<u32>){
  let N=u(1);
  let rows=u(0)/N;
  let r=wid.x+wid.y*65535u;
  let live=r<rows;
  let row=min(r,rows-1u);
  let base=row*N;
  var mx=-3.4e38;
  for(var j=li;j<N;j+=64u){mx=max(mx,a[base+j]);}
  red[li]=mx;workgroupBarrier();
  if(li<32u){red[li]=max(red[li],red[li+32u]);}workgroupBarrier();
  if(li<16u){red[li]=max(red[li],red[li+16u]);}workgroupBarrier();
  if(li<8u){red[li]=max(red[li],red[li+8u]);}workgroupBarrier();
  if(li==0u){red[0]=max(max(max(red[0],red[4]),max(red[2],red[6])),max(max(red[1],red[5]),max(red[3],red[7])));}
  workgroupBarrier();
  let top=red[0];
  workgroupBarrier();
  var s=0.0;
  for(var j=li;j<N;j+=64u){s+=exp(a[base+j]-top);}
  red[li]=s;workgroupBarrier();
  if(li<32u){red[li]+=red[li+32u];}workgroupBarrier();
  if(li<16u){red[li]+=red[li+16u];}workgroupBarrier();
  if(li<8u){red[li]+=red[li+8u];}workgroupBarrier();
  if(li==0u){red[0]=((red[0]+red[4])+(red[2]+red[6]))+((red[1]+red[5])+(red[3]+red[7]));}
  workgroupBarrier();
  let inv=1.0/red[0];
  if(!live){return;}
  for(var j=li;j<N;j+=64u){o[base+j]=exp(a[base+j]-top)*inv;}
}`;
  }

  ipaLogitsTiledSource() {
    return `${params}
@group(0) @binding(1) var<storage,read> a:array<f32>;
@group(0) @binding(2) var<storage,read> b:array<f32>;
@group(0) @binding(3) var<storage,read> c:array<f32>;
@group(0) @binding(4) var<storage,read> d:array<f32>;
@group(0) @binding(5) var<storage,read> e:array<f32>;
@group(0) @binding(6) var<storage,read> f:array<f32>;
@group(0) @binding(7) var<storage,read> g:array<f32>;
@group(0) @binding(8) var<storage,read_write> o:array<f32>;
var<workgroup> sq:array<f32,1024>;
var<workgroup> sk:array<f32,1024>;
var<workgroup> sqp:array<f32,192>;
var<workgroup> skp:array<f32,192>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) wid:vec3<u32>){
  let L=u(1);let H=u(2);let K=u(3);let Q=u(4);let V=u(5);
  let tiles=(L+7u)/8u;
  let tilesPerH=tiles*tiles;
  let flat=wid.x+wid.y*65535u;
  let liveWg=flat<H*tilesPerH;
  let safeWg=min(flat,H*tilesPerH-1u);
  let h=safeWg/tilesPerH;
  let rem=safeWg%tilesPerH;
  let i0=(rem/tiles)*8u;
  let j0=(rem%tiles)*8u;
  for(var t=lane;t<8u*K;t+=64u){
    let r=t/K;let kk=t%K;
    var qv=0.0;if(liveWg && i0+r<L){qv=a[((i0+r)*H+h)*K+kk];}
    var kv=0.0;if(liveWg && j0+r<L){kv=b[((j0+r)*H+h)*2u*K+kk];}
    sq[r*128u+kk]=qv;
    sk[r*128u+kk]=kv;
  }
  let Q3=Q*3u;
  for(var t=lane;t<8u*Q3;t+=64u){
    let r=t/Q3;let pp=t%Q3;
    var qpv=0.0;if(liveWg && i0+r<L){qpv=c[((i0+r)*H+h)*Q3+pp];}
    var kpv=0.0;if(liveWg && j0+r<L){kpv=d[((j0+r)*H+h)*(Q+V)*3u+pp];}
    sqp[r*24u+pp]=qpv;
    skp[r*24u+pp]=kpv;
  }
  workgroupBarrier();
  let ti=lane/8u;let tj=lane%8u;
  let i=i0+ti;let j=j0+tj;
  if(!liveWg || i>=L || j>=L){return;}
  var dotVal=0.0;
  for(var k=0u;k<K;k++){dotVal+=sq[ti*128u+k]*sk[tj*128u+k];}
  var dist=0.0;
  let weight=log(1.0+exp(f[h]))*sqrt(1.0/(3.0*(f32(Q)*4.5)));
  for(var p=0u;p<Q;p++){
    var v=0.0;
    for(var k=0u;k<3u;k++){
      let diff=sqp[ti*24u+p*3u+k]-skp[tj*24u+p*3u+k];
      v+=diff*diff;
    }
    dist+=v*weight;
  }
  o[(h*L+i)*L+j]=dotVal*sqrt(1.0/(3.0*f32(K)))+e[(i*L+j)*H+h]*sqrt(1.0/3.0)-0.5*dist+100000.0*(g[i]*g[j]-1.0);
}`;
  }

  _weightReadHelper(dtype) {
    const half = dtype === "F16" && this.nativeF16;
    const expr =
      dtype === "F32"
        ? "bitcast<f32>(b[index])"
        : dtype === "F16"
          ? half
            ? "f32(b[index])"
            : "select(unpack2x16float(b[index/2u]).x,unpack2x16float(b[index/2u]).y,(index&1u)==1u)"
          : "f32(bitcast<i32>(((b[index/4u]>>((index&3u)*8u))&255u)<<24u)>>24u)*d[row]";
    return `fn w(row:u32,k:u32,K:u32)->f32{let index=row*K+k;return ${expr};}`;
  }

  linear32x64Source(dtype) {
    const half = dtype === "F16" && this.nativeF16;
    const EB = half ? "f16" : "f32";
    const scaleDecl =
      dtype === "I8" ? "@group(0) @binding(4) var<storage,read> d:array<f32>;" : "";
    const unroll = Array.from(
      { length: 16 },
      (_, kk) => `{
      let av=ta[${kk * 17}u+tm2];
      let bv=vec4<f32>(tb[${kk * 17}u+tx]);
      acc0=fma(vec4<f32>(av.x),bv,acc0);
      acc1=fma(vec4<f32>(av.y),bv,acc1);
    }`,
    ).join("\n    ");
    const loadB =
      dtype === "F32" || half
        ? `{
      let c4=li/16u;let kk=li%16u;let k=k0+kk;let col0=n0+c4*4u;
      var bv=vec4<${EB}>(0.0);
      if(inTile && k<K){
        let b0=col0*K+k;
        if(col0+3u<N){
          bv=vec4<${EB}>(${half ? "b[b0],b[b0+K],b[b0+2u*K],b[b0+3u*K]" : "bitcast<f32>(b[b0]),bitcast<f32>(b[b0+K]),bitcast<f32>(b[b0+2u*K]),bitcast<f32>(b[b0+3u*K])"});
        } else {
          if(col0<N){bv.x=${half ? "b[b0]" : "bitcast<f32>(b[b0])"};}
          if(col0+1u<N){bv.y=${half ? "b[b0+K]" : "bitcast<f32>(b[b0+K])"};}
          if(col0+2u<N){bv.z=${half ? "b[b0+2u*K]" : "bitcast<f32>(b[b0+2u*K])"};}
          if(col0+3u<N){bv.w=${half ? "b[b0+3u*K]" : "bitcast<f32>(b[b0+3u*K])"};}
        }
      }
      tb[kk*17u+c4]=bv;
    }`
        : dtype === "F16"
          ? `if((K&1u)==0u){
      if(li<128u){
        let c4=li/8u;let kk=(li%8u)*2u;let col0=n0+c4*4u;
        var lo=vec4<f32>(0.0);var hi=vec4<f32>(0.0);
        if(inTile && k0+kk<K){
          let hasHi=k0+kk+1u<K;let kh=K/2u;let bOff=(col0*K+k0+kk)/2u;
          if(col0+3u<N){
            let p0=unpack2x16float(b[bOff]);let p1=unpack2x16float(b[bOff+kh]);let p2=unpack2x16float(b[bOff+2u*kh]);let p3=unpack2x16float(b[bOff+3u*kh]);
            lo=vec4<f32>(p0.x,p1.x,p2.x,p3.x);if(hasHi){hi=vec4<f32>(p0.y,p1.y,p2.y,p3.y);}
          } else {
            if(col0<N){let p0=unpack2x16float(b[bOff]);lo.x=p0.x;if(hasHi){hi.x=p0.y;}}
            if(col0+1u<N){let p1=unpack2x16float(b[bOff+kh]);lo.y=p1.x;if(hasHi){hi.y=p1.y;}}
            if(col0+2u<N){let p2=unpack2x16float(b[bOff+2u*kh]);lo.z=p2.x;if(hasHi){hi.z=p2.y;}}
            if(col0+3u<N){let p3=unpack2x16float(b[bOff+3u*kh]);lo.w=p3.x;if(hasHi){hi.w=p3.y;}}
          }
        }
        tb[kk*17u+c4]=lo;tb[(kk+1u)*17u+c4]=hi;
      }
    } else {
      let c4=li/16u;let kk=li%16u;let k=k0+kk;let col0=n0+c4*4u;var bv=vec4<f32>(0.0);
      if(inTile && k<K){
        if(col0<N){bv.x=w(col0,k,K);}
        if(col0+1u<N){bv.y=w(col0+1u,k,K);}
        if(col0+2u<N){bv.z=w(col0+2u,k,K);}
        if(col0+3u<N){bv.w=w(col0+3u,k,K);}
      }
      tb[kk*17u+c4]=bv;
    }`
          : `if((K&3u)==0u){
      if(li<64u){
        let c4=li/4u;let kk=(li%4u)*4u;let col0=n0+c4*4u;
        var v0=vec4<f32>(0.0);var v1=vec4<f32>(0.0);var v2=vec4<f32>(0.0);var v3=vec4<f32>(0.0);
        if(inTile && k0+kk<K){
          let kq=K/4u;let bOff=(col0*K+k0+kk)/4u;
          if(col0<N){let word=b[bOff];let sc=d[col0];v0.x=f32(bitcast<i32>(word<<24u)>>24u)*sc;if(k0+kk+1u<K){v1.x=f32(bitcast<i32>(word<<16u)>>24u)*sc;}if(k0+kk+2u<K){v2.x=f32(bitcast<i32>(word<<8u)>>24u)*sc;}if(k0+kk+3u<K){v3.x=f32(bitcast<i32>(word)>>24u)*sc;}}
          if(col0+1u<N){let word=b[bOff+kq];let sc=d[col0+1u];v0.y=f32(bitcast<i32>(word<<24u)>>24u)*sc;if(k0+kk+1u<K){v1.y=f32(bitcast<i32>(word<<16u)>>24u)*sc;}if(k0+kk+2u<K){v2.y=f32(bitcast<i32>(word<<8u)>>24u)*sc;}if(k0+kk+3u<K){v3.y=f32(bitcast<i32>(word)>>24u)*sc;}}
          if(col0+2u<N){let word=b[bOff+2u*kq];let sc=d[col0+2u];v0.z=f32(bitcast<i32>(word<<24u)>>24u)*sc;if(k0+kk+1u<K){v1.z=f32(bitcast<i32>(word<<16u)>>24u)*sc;}if(k0+kk+2u<K){v2.z=f32(bitcast<i32>(word<<8u)>>24u)*sc;}if(k0+kk+3u<K){v3.z=f32(bitcast<i32>(word)>>24u)*sc;}}
          if(col0+3u<N){let word=b[bOff+3u*kq];let sc=d[col0+3u];v0.w=f32(bitcast<i32>(word<<24u)>>24u)*sc;if(k0+kk+1u<K){v1.w=f32(bitcast<i32>(word<<16u)>>24u)*sc;}if(k0+kk+2u<K){v2.w=f32(bitcast<i32>(word<<8u)>>24u)*sc;}if(k0+kk+3u<K){v3.w=f32(bitcast<i32>(word)>>24u)*sc;}}
        }
        tb[kk*17u+c4]=v0;tb[(kk+1u)*17u+c4]=v1;tb[(kk+2u)*17u+c4]=v2;tb[(kk+3u)*17u+c4]=v3;
      }
    } else {
      let c4=li/16u;let kk=li%16u;let k=k0+kk;let col0=n0+c4*4u;var bv=vec4<f32>(0.0);
      if(inTile && k<K){
        if(col0<N){bv.x=w(col0,k,K);}
        if(col0+1u<N){bv.y=w(col0+1u,k,K);}
        if(col0+2u<N){bv.z=w(col0+2u,k,K);}
        if(col0+3u<N){bv.w=w(col0+3u,k,K);}
      }
      tb[kk*17u+c4]=bv;
    }`;
    return `${half ? "enable f16;\n" : ""}${params}
@group(0) @binding(1) var<storage,read> a:array<f32>;
@group(0) @binding(2) var<storage,read> b:array<${half ? "f16" : "u32"}>;
@group(0) @binding(3) var<storage,read> c:array<f32>;
${scaleDecl}
@group(0) @binding(5) var<storage,read_write> o:array<f32>;
${this._weightReadHelper(dtype)}
var<workgroup> ta:array<vec2<f32>,272>;
var<workgroup> tb:array<vec4<${EB}>,272>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li:u32,@builtin(workgroup_id) wid:vec3<u32>){
  let M=u(1);let K=u(2);let N=u(3);let doRelu=u(4)>0u;
  let tilesN=(N+63u)/64u;
  let tilesM=(M+31u)/32u;
  let flat=wid.x+wid.y*65535u;
  let inTile=flat<tilesM*tilesN;
  let tile=min(flat,tilesM*tilesN-1u);
  let m0=(tile/tilesN)*32u;
  let n0=(tile%tilesN)*64u;
  let tm2=li/16u;let tx=li%16u;
  var acc0=vec4<f32>(0.0);var acc1=vec4<f32>(0.0);
  let kkA=li%16u;let r2A=li/16u;let rowA0=m0+r2A*2u;let fullA=rowA0+1u<M;
  for(var k0=0u;k0<K;k0+=16u){
    {
      var av=vec2<f32>(0.0);
      if(inTile && k0+kkA<K){
        let aOff=rowA0*K+k0+kkA;
        if(fullA){av=vec2<f32>(a[aOff],a[aOff+K]);}
        else if(rowA0<M){av.x=a[aOff];}
      }
      ta[kkA*17u+r2A]=av;
    }
    ${loadB}
    workgroupBarrier();
    ${unroll}
    workgroupBarrier();
  }
  if(!inTile){return;}
  let colBase=n0+tx*4u;
  var bVec=vec4<f32>(0.0);
  if(colBase<N){bVec.x=c[colBase];}
  if(colBase+1u<N){bVec.y=c[colBase+1u];}
  if(colBase+2u<N){bVec.z=c[colBase+2u];}
  if(colBase+3u<N){bVec.w=c[colBase+3u];}
  let r0=m0+tm2*2u;
  if(r0<M){
    var res=acc0+bVec;
    if(doRelu){res=max(res,vec4<f32>(0.0));}
    if(colBase<N){o[r0*N+colBase]=res.x;}
    if(colBase+1u<N){o[r0*N+colBase+1u]=res.y;}
    if(colBase+2u<N){o[r0*N+colBase+2u]=res.z;}
    if(colBase+3u<N){o[r0*N+colBase+3u]=res.w;}
  }
  let r1=r0+1u;
  if(r1<M){
    var res=acc1+bVec;
    if(doRelu){res=max(res,vec4<f32>(0.0));}
    if(colBase<N){o[r1*N+colBase]=res.x;}
    if(colBase+1u<N){o[r1*N+colBase+1u]=res.y;}
    if(colBase+2u<N){o[r1*N+colBase+2u]=res.z;}
    if(colBase+3u<N){o[r1*N+colBase+3u]=res.w;}
  }
}`;
  }

  linear64x128Source(dtype) {
    const half = dtype === "F16" && this.nativeF16;
    const EB = half ? "f16" : "f32";
    const scaleDecl =
      dtype === "I8" ? "@group(0) @binding(4) var<storage,read> d:array<f32>;" : "";
    const unroll = Array.from(
      { length: 16 },
      (_, kk) => `{
      let av=ta[${kk * 17}u+tm4];
      let bv0=vec4<f32>(tb[${kk * 33}u+tx]);
      let bv1=vec4<f32>(tb[${kk * 33 + 16}u+tx]);
      acc00=fma(vec4<f32>(av.x),bv0,acc00);acc01=fma(vec4<f32>(av.x),bv1,acc01);
      acc10=fma(vec4<f32>(av.y),bv0,acc10);acc11=fma(vec4<f32>(av.y),bv1,acc11);
      acc20=fma(vec4<f32>(av.z),bv0,acc20);acc21=fma(vec4<f32>(av.z),bv1,acc21);
      acc30=fma(vec4<f32>(av.w),bv0,acc30);acc31=fma(vec4<f32>(av.w),bv1,acc31);
    }`,
    ).join("\n    ");
    const loadB =
      dtype === "F16" && !half
        ? `if((K&1u)==0u){
      let kk=(li%8u)*2u;let c4=li/8u;let col0=n0+c4*4u;
      var lo=vec4<f32>(0.0);var hi=vec4<f32>(0.0);
      if(inTile && k0+kk<K){
        let hasHi=k0+kk+1u<K;let kh=K/2u;let bOff=(col0*K+k0+kk)/2u;
        if(col0+3u<N){
          let p0=unpack2x16float(b[bOff]);let p1=unpack2x16float(b[bOff+kh]);let p2=unpack2x16float(b[bOff+2u*kh]);let p3=unpack2x16float(b[bOff+3u*kh]);
          lo=vec4<f32>(p0.x,p1.x,p2.x,p3.x);if(hasHi){hi=vec4<f32>(p0.y,p1.y,p2.y,p3.y);}
        } else {
          if(col0<N){let p0=unpack2x16float(b[bOff]);lo.x=p0.x;if(hasHi){hi.x=p0.y;}}
          if(col0+1u<N){let p1=unpack2x16float(b[bOff+kh]);lo.y=p1.x;if(hasHi){hi.y=p1.y;}}
          if(col0+2u<N){let p2=unpack2x16float(b[bOff+2u*kh]);lo.z=p2.x;if(hasHi){hi.z=p2.y;}}
          if(col0+3u<N){let p3=unpack2x16float(b[bOff+3u*kh]);lo.w=p3.x;if(hasHi){hi.w=p3.y;}}
        }
      }
      tb[kk*33u+c4]=lo;tb[(kk+1u)*33u+c4]=hi;
    } else {
      for(var e=0u;e<2u;e++){
        let slot=li+e*256u;let kk=slot%16u;let c4=slot/16u;let k=k0+kk;let col0=n0+c4*4u;var bv=vec4<f32>(0.0);
        if(inTile && k<K){
          if(col0<N){bv.x=w(col0,k,K);}
          if(col0+1u<N){bv.y=w(col0+1u,k,K);}
          if(col0+2u<N){bv.z=w(col0+2u,k,K);}
          if(col0+3u<N){bv.w=w(col0+3u,k,K);}
        }
        tb[kk*33u+c4]=bv;
      }
    }`
        : dtype === "I8"
          ? `if((K&3u)==0u){
      if(li<128u){
        let c4=li/4u;let kk=(li%4u)*4u;let col0=n0+c4*4u;
        var v0=vec4<f32>(0.0);var v1=vec4<f32>(0.0);var v2=vec4<f32>(0.0);var v3=vec4<f32>(0.0);
        if(inTile && k0+kk<K){
          let kq=K/4u;let bOff=(col0*K+k0+kk)/4u;
          if(col0<N){let word=b[bOff];let sc=d[col0];v0.x=f32(bitcast<i32>(word<<24u)>>24u)*sc;if(k0+kk+1u<K){v1.x=f32(bitcast<i32>(word<<16u)>>24u)*sc;}if(k0+kk+2u<K){v2.x=f32(bitcast<i32>(word<<8u)>>24u)*sc;}if(k0+kk+3u<K){v3.x=f32(bitcast<i32>(word)>>24u)*sc;}}
          if(col0+1u<N){let word=b[bOff+kq];let sc=d[col0+1u];v0.y=f32(bitcast<i32>(word<<24u)>>24u)*sc;if(k0+kk+1u<K){v1.y=f32(bitcast<i32>(word<<16u)>>24u)*sc;}if(k0+kk+2u<K){v2.y=f32(bitcast<i32>(word<<8u)>>24u)*sc;}if(k0+kk+3u<K){v3.y=f32(bitcast<i32>(word)>>24u)*sc;}}
          if(col0+2u<N){let word=b[bOff+2u*kq];let sc=d[col0+2u];v0.z=f32(bitcast<i32>(word<<24u)>>24u)*sc;if(k0+kk+1u<K){v1.z=f32(bitcast<i32>(word<<16u)>>24u)*sc;}if(k0+kk+2u<K){v2.z=f32(bitcast<i32>(word<<8u)>>24u)*sc;}if(k0+kk+3u<K){v3.z=f32(bitcast<i32>(word)>>24u)*sc;}}
          if(col0+3u<N){let word=b[bOff+3u*kq];let sc=d[col0+3u];v0.w=f32(bitcast<i32>(word<<24u)>>24u)*sc;if(k0+kk+1u<K){v1.w=f32(bitcast<i32>(word<<16u)>>24u)*sc;}if(k0+kk+2u<K){v2.w=f32(bitcast<i32>(word<<8u)>>24u)*sc;}if(k0+kk+3u<K){v3.w=f32(bitcast<i32>(word)>>24u)*sc;}}
        }
        tb[kk*33u+c4]=v0;tb[(kk+1u)*33u+c4]=v1;tb[(kk+2u)*33u+c4]=v2;tb[(kk+3u)*33u+c4]=v3;
      }
    } else {
      for(var e=0u;e<2u;e++){
        let slot=li+e*256u;let kk=slot%16u;let c4=slot/16u;let k=k0+kk;let col0=n0+c4*4u;var bv=vec4<f32>(0.0);
        if(inTile && k<K){
          if(col0<N){bv.x=w(col0,k,K);}
          if(col0+1u<N){bv.y=w(col0+1u,k,K);}
          if(col0+2u<N){bv.z=w(col0+2u,k,K);}
          if(col0+3u<N){bv.w=w(col0+3u,k,K);}
        }
        tb[kk*33u+c4]=bv;
      }
    }`
          : `for(var e=0u;e<2u;e++){
      let slot=li+e*256u;let kk=slot%16u;let c4=slot/16u;let k=k0+kk;let col0=n0+c4*4u;
      var bv=vec4<${EB}>(0.0);
      if(inTile && k<K){
        let b0=col0*K+k;
        if(col0+3u<N){
          bv=vec4<${EB}>(${half ? "b[b0],b[b0+K],b[b0+2u*K],b[b0+3u*K]" : "bitcast<f32>(b[b0]),bitcast<f32>(b[b0+K]),bitcast<f32>(b[b0+2u*K]),bitcast<f32>(b[b0+3u*K])"});
        } else {
          if(col0<N){bv.x=${half ? "b[b0]" : "bitcast<f32>(b[b0])"};}
          if(col0+1u<N){bv.y=${half ? "b[b0+K]" : "bitcast<f32>(b[b0+K])"};}
          if(col0+2u<N){bv.z=${half ? "b[b0+2u*K]" : "bitcast<f32>(b[b0+2u*K])"};}
          if(col0+3u<N){bv.w=${half ? "b[b0+3u*K]" : "bitcast<f32>(b[b0+3u*K])"};}
        }
      }
      tb[kk*33u+c4]=bv;
    }`;
    const storeRows = [0, 1, 2, 3]
      .map(
        (i) => `{
      let m=m0+tm4*4u+${i}u;
      if(m<M){
        var r0=acc${i}0+b0;var r1=acc${i}1+b1;
        if(doRelu){r0=max(r0,vec4<f32>(0.0));r1=max(r1,vec4<f32>(0.0));}
        let rowOff=m*N;
        if(c0<N){o[rowOff+c0]=r0.x;} if(c0+1u<N){o[rowOff+c0+1u]=r0.y;} if(c0+2u<N){o[rowOff+c0+2u]=r0.z;} if(c0+3u<N){o[rowOff+c0+3u]=r0.w;}
        if(c1<N){o[rowOff+c1]=r1.x;} if(c1+1u<N){o[rowOff+c1+1u]=r1.y;} if(c1+2u<N){o[rowOff+c1+2u]=r1.z;} if(c1+3u<N){o[rowOff+c1+3u]=r1.w;}
      }
    }`,
      )
      .join("\n    ");
    return `${half ? "enable f16;\n" : ""}${params}
@group(0) @binding(1) var<storage,read> a:array<f32>;
@group(0) @binding(2) var<storage,read> b:array<${half ? "f16" : "u32"}>;
@group(0) @binding(3) var<storage,read> c:array<f32>;
${scaleDecl}
@group(0) @binding(5) var<storage,read_write> o:array<f32>;
${this._weightReadHelper(dtype)}
var<workgroup> ta:array<vec4<f32>,272>;
var<workgroup> tb:array<vec4<${EB}>,528>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li:u32,@builtin(workgroup_id) wid:vec3<u32>){
  let M=u(1);let K=u(2);let N=u(3);let doRelu=u(4)>0u;
  let tilesN=(N+127u)/128u;
  let tilesM=(M+63u)/64u;
  let flat=wid.x+wid.y*65535u;
  let inTile=flat<tilesM*tilesN;
  let tile=min(flat,tilesM*tilesN-1u);
  let m0=(tile/tilesN)*64u;
  let n0=(tile%tilesN)*128u;
  let tm4=li/16u;let tx=li%16u;
  var acc00=vec4<f32>(0.0);var acc01=vec4<f32>(0.0);
  var acc10=vec4<f32>(0.0);var acc11=vec4<f32>(0.0);
  var acc20=vec4<f32>(0.0);var acc21=vec4<f32>(0.0);
  var acc30=vec4<f32>(0.0);var acc31=vec4<f32>(0.0);
  let kkA=li%16u;let r4A=li/16u;let rowA0=m0+r4A*4u;let fullA=rowA0+3u<M;
  for(var k0=0u;k0<K;k0+=16u){
    {
      var av=vec4<f32>(0.0);
      if(inTile && k0+kkA<K){
        let aOff=rowA0*K+k0+kkA;
        if(fullA){av=vec4<f32>(a[aOff],a[aOff+K],a[aOff+2u*K],a[aOff+3u*K]);}
        else {
          if(rowA0<M){av.x=a[aOff];}
          if(rowA0+1u<M){av.y=a[aOff+K];}
          if(rowA0+2u<M){av.z=a[aOff+2u*K];}
          if(rowA0+3u<M){av.w=a[aOff+3u*K];}
        }
      }
      ta[kkA*17u+r4A]=av;
    }
    ${loadB}
    workgroupBarrier();
    ${unroll}
    workgroupBarrier();
  }
  if(!inTile){return;}
  let c0=n0+tx*4u;let c1=c0+64u;
  var b0=vec4<f32>(0.0);var b1=vec4<f32>(0.0);
  if(c0<N){b0.x=c[c0];} if(c0+1u<N){b0.y=c[c0+1u];} if(c0+2u<N){b0.z=c[c0+2u];} if(c0+3u<N){b0.w=c[c0+3u];}
  if(c1<N){b1.x=c[c1];} if(c1+1u<N){b1.y=c[c1+1u];} if(c1+2u<N){b1.z=c[c1+2u];} if(c1+3u<N){b1.w=c[c1+3u];}
  ${storeRows}
}`;
  }

  async finish(outputs) {
    let staging = null;
    try {
      this._endPass();
      const sizes = outputs.map((t) => Math.max(4, product(t.shape) * 4));
      const totalBytes = sizes.reduce((a, b) => a + b, 0);
      staging = this.buffer(
        totalBytes,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      );
      let off = 0;
      for (let i = 0; i < outputs.length; i++) {
        this.encoder.copyBufferToBuffer(outputs[i].buffer, 0, staging, off, sizes[i]);
        off += sizes[i];
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
      this._drainPendingDestroy();
      await staging.mapAsync(GPUMapMode.READ);
      const mapped = staging.getMappedRange();
      const result = [];
      let pos = 0;
      for (let i = 0; i < outputs.length; i++) {
        const len = product(outputs[i].shape) * 4;
        result.push(new Float32Array(mapped.slice(pos, pos + len)));
        pos += sizes[i];
      }
      return result;
    } finally {
      if (staging) {
        try {
          staging.unmap();
        } catch {}
        staging.destroy();
      }
      this._endPass();
      this.encoder = null;
      this.pinned.clear();
      this.collect();
    }
  }
  cancel() {
    this._endPass();
    this.encoder = null;
    this.pinned.clear();
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
    this._endPass();
    this.encoder = null;
    for (const t of this.live) t.buffer.destroy();
    this.live.clear();
    for (const t of this.pool) t.buffer.destroy();
    for (const b of this.pendingDestroy) b.destroy();
    for (const t of Object.values(this.weights)) {
      t.buffer.destroy();
      t.scales?.destroy();
    }
    this.parameterBuffer.destroy();
    this.pool = [];
    this.pendingDestroy = [];
    this.weights = {};
    this.weightBytes = 0;
    this.scratchBytes = 0;
    this.pipelines.clear();
  }
}
