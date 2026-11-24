import { formatGpuLossDetail } from './resources.js';

// Eager WGSL tensor operations. All activation arithmetic stays on the GPU.
// Scratch buffers are reused only after their last encoded use; host uploads are
// never pooled, so queue.writeBuffer cannot overwrite earlier pending commands.
const DEFAULT_MAX_POOL_BYTES = 256 * 1024 * 1024;

export class GPUOps {
  constructor(resources, { maxPoolBytes = DEFAULT_MAX_POOL_BYTES } = {}) {
    this.owner = resources;
    this.device = resources.device;
    this.pipelines = new Map();
    this.pool = new Map();
    this.poolBytes = 0;
    this.maxPoolBytes = maxPoolBytes;
    this.deferredReleases = [];
    this.tensors = new Set();
    this.scopes = [];
    this.encoder = null;
    this.pass = null;
  }
  begin() {
    if (this.encoder) throw new Error('GPU command transaction already active');
    this.encoder = this.device.createCommandEncoder();
    this.pass = null;
  }
  endPass() {
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
  }
  releaseBuffer(buffer) {
    if (this.encoder) this.deferredReleases.push(buffer);
    else this.owner.release(buffer);
  }
  drainDeferredReleases() {
    while (this.deferredReleases.length) {
      this.owner.release(this.deferredReleases.pop());
    }
  }
  evictOnePooled() {
    for (const [size, bucket] of this.pool.entries()) {
      if (bucket.length) {
        const buf = bucket.pop();
        this.poolBytes -= buf.size;
        if (!bucket.length) this.pool.delete(size);
        this.releaseBuffer(buf);
        return true;
      }
      this.pool.delete(size);
    }
    return false;
  }
  tensor(length, host = false) {
    if (!Number.isSafeInteger(length) || length < 1 || length * 4 > this.device.limits.maxStorageBufferBindingSize)
      throw new RangeError('Tensor exceeds the negotiated binding limit');
    const size = length * 4;
    const bucket = !host ? this.pool.get(size) : undefined;
    let buffer;
    if (bucket?.length) {
      buffer = bucket.pop();
      this.poolBytes -= buffer.size;
      if (!bucket.length) this.pool.delete(size);
    } else {
      if (!host && size <= this.maxPoolBytes) {
        while (this.poolBytes + size > this.maxPoolBytes) {
          if (!this.evictOnePooled()) break;
        }
      }
      buffer = this.owner.allocate(size, 128 | 8 | 4, 'RFD3 activation');
    }
    const t = { buffer, length, host, released: false };
    this.tensors.add(t);
    for (const scope of this.scopes) scope.push(t);
    return t;
  }
  upload(data) {
    const t = this.tensor(data.length, true);
    this.device.queue.writeBuffer(t.buffer, 0, Float32Array.from(data));
    return t;
  }
  free(t) {
    if (!t || t.released || !this.tensors.delete(t)) return;
    t.released = true;
    if (t.host || t.buffer.size > this.maxPoolBytes || this.maxPoolBytes <= 0) {
      this.releaseBuffer(t.buffer);
      return;
    }
    while (this.poolBytes + t.buffer.size > this.maxPoolBytes) {
      if (!this.evictOnePooled()) break;
    }
    if (this.poolBytes + t.buffer.size <= this.maxPoolBytes) {
      let b = this.pool.get(t.buffer.size);
      if (!b) this.pool.set(t.buffer.size, b = []);
      b.push(t.buffer);
      this.poolBytes += t.buffer.size;
    } else {
      this.releaseBuffer(t.buffer);
    }
  }
  scope(fn) {
    const created = [];
    this.scopes.push(created);
    try {
      const result = fn(), keep = new Set();
      const walk = v => {
        if (!v || typeof v !== 'object') return;
        if (v.buffer && v.length !== undefined) { keep.add(v); return; }
        for (const x of Object.values(v)) walk(x);
      };
      walk(result);
      for (const t of created) if (!keep.has(t)) this.free(t);
      return result;
    } catch (error) {
      for (const t of created) this.free(t);
      throw error;
    } finally {
      this.scopes.pop();
    }
  }
  dispatch(code, buffers, groups, label = 'RFD3 op') {
    if (!this.encoder) throw new Error('No GPU command transaction');
    let pipeline = this.pipelines.get(code);
    if (!pipeline) {
      const module = this.device.createShaderModule({ code, label });
      pipeline = this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' }, label });
      this.pipelines.set(code, pipeline);
    }
    const group = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((t, binding) => ({ binding, resource: { buffer: t.buffer ?? t } })),
    });
    if (!this.pass) this.pass = this.encoder.beginComputePass({ label });
    this.pass.setPipeline(pipeline);
    this.pass.setBindGroup(0, group);
    this.pass.dispatchWorkgroups(...groups);
  }
  flush() {
    if (!this.encoder) return;
    this.endPass();
    const encoder = this.encoder;
    this.encoder = null;
    this.device.queue.submit([encoder.finish()]);
    this.drainDeferredReleases();
    this.encoder = this.device.createCommandEncoder();
  }
  abort() {
    try { this.endPass(); } catch { /* ignore */ }
    this.encoder = null;
    this.pass = null;
    this.drainDeferredReleases();
  }
  kernel(inputs, length, body, label = 'RFD3 elementwise') {
    const out = this.tensor(length), num = Math.ceil(length / 64), x = Math.min(65535, num), y = Math.ceil(num / x);
    const code = inputs.map((_, i) => `@group(0) @binding(${i}) var<storage,read> a${i}:array<f32>;`).join('\n') + `
@group(0) @binding(${inputs.length}) var<storage,read_write> out:array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){let idx=gid.x+gid.y*${x * 64}u;if(idx>=${length}u){return;}${body}}`;
    this.dispatch(code, [...inputs, out], [x, y], label);
    return out;
  }
  fill(length, value = 0) {
    return this.kernel([], length, `out[idx]=${Number(value).toFixed(12)};`);
  }
  add(a, b) {
    if (a.length !== b.length) throw new Error('GPU add shape mismatch');
    return this.kernel([a, b], a.length, 'out[idx]=a0[idx]+a1[idx];');
  }
  mul(a, b) {
    return this.kernel([a, b], a.length, 'out[idx]=a0[idx]*a1[idx];');
  }
  activation(a, kind) {
    const expression = { relu: 'max(v,0.0)', sigmoid: '1.0/(1.0+exp(-v))', silu: 'v/(1.0+exp(-v))' }[kind];
    if (!expression) throw new Error('Unknown activation');
    return this.kernel([a], a.length, `let v=a0[idx];out[idx]=${expression};`);
  }
  linear(a, w, bias, { reluInput = false } = {}) {
    const [n, k] = w.shape, m = a.length / k;
    if (!Number.isInteger(m)) throw new Error('GPU linear shape mismatch');
    const out = this.tensor(m * n), dtype = { F32: 0, F16: 1, I8: 2 }[w.dtype], inputs = [a, w];
    let scale = '';
    if (dtype === 2) { inputs.push(w.scales); scale = '*scales[row]'; }
    if (bias) inputs.push(bias);
    const code = `@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> w:array<u32>;
${dtype === 2 ? '@group(0) @binding(2) var<storage,read> scales:array<f32>;' : ''}
${bias ? `@group(0) @binding(${inputs.length - 1}) var<storage,read> bias:array<f32>;` : ''}
@group(0) @binding(${inputs.length}) var<storage,read_write> out:array<f32>;
var<workgroup> tileA:array<f32,512>;
var<workgroup> tileB:array<f32,512>;
fn weight(i:u32,row:u32)->f32 {${dtype === 0 ? 'return bitcast<f32>(w[i]);' : dtype === 1 ? 'return unpack2x16float(w[i/2u])[i%2u];' : `return f32(bitcast<i32>(((w[i/4u]>>(8u*(i%4u)))&255u)<<24u)>>24)${scale};`}}
@compute @workgroup_size(16,16) fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_id) id:vec3<u32>){
 let tileRow=(g.y+g.z*65535u)*32u;let tileCol=g.x*32u;
 let row0=tileRow+id.y*2u;let row1=row0+1u;
 let col0=tileCol+id.x*2u;let col1=col0+1u;
 let tid=id.y*16u+id.x;
 var s00=0.0;var s01=0.0;var s10=0.0;var s11=0.0;
 for(var base=0u;base<${k}u;base+=16u){
  for(var t=0u;t<2u;t++){
   let idx=tid+t*256u;let r=idx/16u;let kk=idx%16u;
   let gr=tileRow+r;let gc=tileCol+r;let gk=base+kk;
   var av=0.0;var bv=0.0;
   if(gr<${m}u&&gk<${k}u){av=${reluInput ? `max(x[gr*${k}u+gk],0.0)` : `x[gr*${k}u+gk]`};}
   if(gc<${n}u&&gk<${k}u){bv=weight(gc*${k}u+gk,gc);}
   tileA[idx]=av;tileB[idx]=bv;
  }
  workgroupBarrier();
  for(var j=0u;j<16u;j++){
   let a0=tileA[(id.y*2u)*16u+j];let a1=tileA[(id.y*2u+1u)*16u+j];
   let b0=tileB[(id.x*2u)*16u+j];let b1=tileB[(id.x*2u+1u)*16u+j];
   s00+=a0*b0;s01+=a0*b1;s10+=a1*b0;s11+=a1*b1;
  }
  workgroupBarrier();
 }
 if(row0<${m}u&&col0<${n}u){out[row0*${n}u+col0]=s00${bias ? '+bias[col0]' : ''};}
 if(row0<${m}u&&col1<${n}u){out[row0*${n}u+col1]=s01${bias ? '+bias[col1]' : ''};}
 if(row1<${m}u&&col0<${n}u){out[row1*${n}u+col0]=s10${bias ? '+bias[col0]' : ''};}
 if(row1<${m}u&&col1<${n}u){out[row1*${n}u+col1]=s11${bias ? '+bias[col1]' : ''};}
}`;
    const groupsY = Math.ceil(m / 32);
    this.dispatch(code, [...inputs, out], [Math.ceil(n / 32), Math.min(65535, groupsY), Math.ceil(groupsY / 65535)], 'RFD3 packed linear');
    return out;
  }
  fusedPosition(pos, mask, wA, bA, wB, bB, { sinusoidal = false, addTo } = {}) {
    const n = pos.length / 3, channels = wA.shape[0], total = n * n * channels;
    const dtypeA = { F32: 0, F16: 1, I8: 2 }[wA.dtype], dtypeB = { F32: 0, F16: 1, I8: 2 }[wB.dtype];
    const inputs = [pos, mask, wA, wB], decls = [
      '@group(0) @binding(0) var<storage,read> pos:array<f32>;',
      '@group(0) @binding(1) var<storage,read> mask:array<f32>;',
      '@group(0) @binding(2) var<storage,read> wA:array<u32>;',
      '@group(0) @binding(3) var<storage,read> wB:array<u32>;',
    ];
    const addBuf = (buf, name) => {
      const i = inputs.length;
      inputs.push(buf);
      decls.push(`@group(0) @binding(${i}) var<storage,read> ${name}:array<f32>;`);
    };
    if (dtypeA === 2) addBuf(wA.scales, 'sA');
    if (bA) addBuf(bA, 'bA');
    if (dtypeB === 2) addBuf(wB.scales, 'sB');
    if (bB) addBuf(bB, 'bB');
    if (addTo) addBuf(addTo, 'addIn');
    const out = this.tensor(total), num = Math.ceil(total / 64), x = Math.min(65535, num), y = Math.ceil(num / x);
    decls.push(`@group(0) @binding(${inputs.length}) var<storage,read_write> out:array<f32>;`);
    const unpack = (wName, sName, dt) =>
      dt === 0 ? `return bitcast<f32>(${wName}[i]);`
      : dt === 1 ? `return unpack2x16float(${wName}[i/2u])[i%2u];`
      : `return f32(bitcast<i32>(((${wName}[i/4u]>>(8u*(i%4u)))&255u)<<24u)>>24)*${sName}[row];`;
    const code = `${decls.join('\n')}
fn weightA(i:u32,row:u32)->f32 {${unpack('wA', 'sA', dtypeA)}}
fn weightB(i:u32,row:u32)->f32 {${unpack('wB', 'sB', dtypeB)}}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 let idx=gid.x+gid.y*${x * 64}u;if(idx>=${total}u){return;}
 let pair=idx/${channels}u;let m=mask[pair];
 if(m==0.0){out[idx]=${addTo ? 'addIn[idx]' : '0.0'};return;}
 let i=pair/${n}u;let j=pair%${n}u;let col=idx%${channels}u;
 var square=0.0;for(var c=0u;c<3u;c++){let d=pos[i*3u+c]-pos[j*3u+c];square+=d*d;}
 ${sinusoidal
   ? `let dist=sqrt(square);var sumA=0.0;for(var k=0u;k<64u;k++){let angle=dist*exp(-log(10000.0)*f32(k%32u)/32.0);sumA+=select(sin(angle),cos(angle),k>=32u)*weightA(col*64u+k,col);}`
   : `var sumA=(1.0/(1.0+max(square,1e-6)))*weightA(col,col);`}
 ${bA ? 'sumA+=bA[col];' : ''}
 var sumB=m*weightB(col,col);
 ${bB ? 'sumB+=bB[col];' : ''}
 let v=(sumA+sumB)*m;
 out[idx]=${addTo ? 'addIn[idx]+v' : 'v'};
}`;
    this.dispatch(code, [...inputs, out], [x, y], 'RFD3 fused position embed');
    return out;
  }
  norm(a, channels, gain) {
    const rows = a.length / channels, out = this.tensor(a.length);
    if (!Number.isInteger(rows)) throw new Error('GPU RMSNorm shape mismatch');
    const code = `@group(0) @binding(0) var<storage,read> a:array<f32>;
${gain ? '@group(0) @binding(1) var<storage,read> gain:array<f32>;' : ''}
@group(0) @binding(${gain ? 2 : 1}) var<storage,read_write> out:array<f32>;
var<workgroup> sums:array<f32,64>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) id:u32){let row=g.x+g.y*65535u;if(row>=${rows}u){return;}var s=0.0;
 for(var c=id;c<${channels}u;c+=64u){let v=a[row*${channels}u+c];s+=v*v;}sums[id]=s;workgroupBarrier();
 for(var step=32u;step>0u;step/=2u){if(id<step){sums[id]+=sums[id+step];}workgroupBarrier();}
 let scale=inverseSqrt(sums[0]/${channels}.0+0.00000011920928955078125);
 for(var c=id;c<${channels}u;c+=64u){out[row*${channels}u+c]=a[row*${channels}u+c]*scale${gain ? '*gain[c]' : ''};}}`;
    this.dispatch(code, gain ? [a, gain, out] : [a, out], [Math.min(rows, 65535), Math.ceil(rows / 65535)], 'RFD3 RMSNorm');
    return out;
  }
  concat(arrays, widths, rows) {
    let offset = 0;
    const width = widths.reduce((a, b) => a + b, 0);
    const clauses = widths.map((w, i) => {
      const start = offset;
      offset += w;
      return `if(c>=${start}u&&c<${offset}u){out[idx]=a${i}[row*${w}u+c-${start}u];}`;
    });
    return this.kernel(arrays, rows * width, `let row=idx/${width}u;let c=idx%${width}u;${clauses.join('')}`);
  }
  knn(coords, forced, rows, k) {
    const out = this.tensor(rows * k);
    const code = `@group(0) @binding(0) var<storage,read> coords:array<f32>;
@group(0) @binding(1) var<storage,read> forced:array<f32>;
@group(0) @binding(2) var<storage,read_write> out:array<f32>;
var<workgroup> wgDistA:array<f32,128>;
var<workgroup> wgIdxA:array<u32,128>;
var<workgroup> wgDistB:array<f32,128>;
var<workgroup> wgIdxB:array<u32,128>;
@compute @workgroup_size(128) fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) id:u32){
 let row=g.x+g.y*65535u;if(row>=${rows}u){return;}
 wgDistA[id]=3.402823e38;wgIdxA[id]=0x7fffff00u+id;
 let rx=coords[row*3u];let ry=coords[row*3u+1u];let rz=coords[row*3u+2u];
 for(var base=0u;base<${rows}u;base+=128u){
  let j=base+id;
  if(j<${rows}u){
   if(forced[row*${rows}u+j]!=0.0){wgDistB[id]= -1.0;}
   else{
    let dx=rx-coords[j*3u];let dy=ry-coords[j*3u+1u];let dz=rz-coords[j*3u+2u];
    wgDistB[id]=dx*dx+dy*dy+dz*dz;
   }
   wgIdxB[id]=j;
  }else{wgDistB[id]=3.402823e38;wgIdxB[id]=j;}
  workgroupBarrier();
  for(var span=2u;span<=128u;span*=2u){
   for(var stride=span/2u;stride>0u;stride/=2u){
    let other=id^stride;
    if(other>id){
     let desc=(id&span)!=0u;
     let di=wgDistB[id];let dj=wgDistB[other];let ii=wgIdxB[id];let ij=wgIdxB[other];
     let less=(di<dj)||(di==dj&&ii<ij);
     if(less==desc){wgDistB[id]=dj;wgDistB[other]=di;wgIdxB[id]=ij;wgIdxB[other]=ii;}
    }
    workgroupBarrier();
   }
  }
  let bd=wgDistB[127u-id];let bi=wgIdxB[127u-id];
  let ad=wgDistA[id];let ai=wgIdxA[id];
  if((bd<ad)||(bd==ad&&bi<ai)){wgDistA[id]=bd;wgIdxA[id]=bi;}
  workgroupBarrier();
  for(var stride=64u;stride>0u;stride/=2u){
   let other=id^stride;
   if(other>id){
    let di=wgDistA[id];let dj=wgDistA[other];let ii=wgIdxA[id];let ij=wgIdxA[other];
    if((dj<di)||(dj==di&&ij<ii)){wgDistA[id]=dj;wgDistA[other]=di;wgIdxA[id]=ij;wgIdxA[other]=ii;}
   }
   workgroupBarrier();
  }
 }
 if(id>=${k}u){wgIdxA[id]=0xffffffffu;}
 workgroupBarrier();
 for(var span=2u;span<=128u;span*=2u){
  for(var stride=span/2u;stride>0u;stride/=2u){
   let other=id^stride;
   if(other>id){
    let desc=(id&span)!=0u;
    let ii=wgIdxA[id];let ij=wgIdxA[other];
    if(ii!=ij&&(ii<ij)==desc){wgIdxA[id]=ij;wgIdxA[other]=ii;}
   }
   workgroupBarrier();
  }
 }
 if(id<${k}u){out[row*${k}u+id]=f32(wgIdxA[id]);}
}`;
    this.dispatch(code, [coords, forced, out], [Math.min(rows, 65535), Math.ceil(rows / 65535)], 'RFD3 KNN128');
    return { idx: out, k };
  }
  multiNeighboursSelect(intra, reserved, rows, k, inter) {
    const out = this.tensor(rows * rows);
    const code = `@group(0) @binding(0) var<storage,read> intra:array<f32>;
@group(0) @binding(1) var<storage,read> reserved:array<f32>;
@group(0) @binding(2) var<storage,read_write> out:array<f32>;
var<workgroup> wgDistA:array<f32,128>;
var<workgroup> wgIdxA:array<u32,128>;
var<workgroup> wgDistB:array<f32,128>;
var<workgroup> wgIdxB:array<u32,128>;
var<workgroup> wgResSum:array<u32,128>;
@compute @workgroup_size(128) fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) id:u32){
 let row=g.x+g.y*65535u;if(row>=${rows}u){return;}
 wgDistA[id]=3.402823e38;wgIdxA[id]=0x7fffff00u+id;
 var rCount=0u;
 for(var base=0u;base<${rows}u;base+=128u){
  let j=base+id;
  if(j<${rows}u){
   let rv=reserved[row*${rows}u+j];
   out[row*${rows}u+j]=rv;
   rCount+=u32(rv);
   wgDistB[id]=intra[row*${rows}u+j];
   wgIdxB[id]=j;
  }else{wgDistB[id]=3.402823e38;wgIdxB[id]=j;}
  workgroupBarrier();
  for(var span=2u;span<=128u;span*=2u){
   for(var stride=span/2u;stride>0u;stride/=2u){
    let other=id^stride;
    if(other>id){
     let desc=(id&span)!=0u;
     let di=wgDistB[id];let dj=wgDistB[other];let ii=wgIdxB[id];let ij=wgIdxB[other];
     let less=(di<dj)||(di==dj&&ii<ij);
     if(less==desc){wgDistB[id]=dj;wgDistB[other]=di;wgIdxB[id]=ij;wgIdxB[other]=ii;}
    }
    workgroupBarrier();
   }
  }
  let bd=wgDistB[127u-id];let bi=wgIdxB[127u-id];
  let ad=wgDistA[id];let ai=wgIdxA[id];
  if((bd<ad)||(bd==ad&&bi<ai)){wgDistA[id]=bd;wgIdxA[id]=bi;}
  workgroupBarrier();
  for(var stride=64u;stride>0u;stride/=2u){
   let other=id^stride;
   if(other>id){
    let di=wgDistA[id];let dj=wgDistA[other];let ii=wgIdxA[id];let ij=wgIdxA[other];
    if((dj<di)||(dj==di&&ij<ii)){wgDistA[id]=dj;wgDistA[other]=di;wgIdxA[id]=ij;wgIdxA[other]=ii;}
   }
   workgroupBarrier();
  }
 }
 wgResSum[id]=rCount;workgroupBarrier();
 for(var s=64u;s>0u;s/=2u){if(id<s){wgResSum[id]+=wgResSum[id+s];}workgroupBarrier();}
 let resTotal=wgResSum[0];
 let col=wgIdxA[id];
 if(col<${rows}u&&(id<${k - inter}u||(id>=${k - inter}u+resTotal&&id<${k}u))){
  out[row*${rows}u+col]+=1.0;
 }
}`;
    this.dispatch(code, [intra, reserved, out], [Math.min(rows, 65535), Math.ceil(rows / 65535)], 'RFD3 multi neighbour select');
    return out;
  }
  maskToKnn(mask, rows, k, clampOne = false) {
    const idx = this.tensor(rows * k), logw = this.tensor(rows * k);
    const num = Math.ceil(rows / 64), x = Math.min(65535, num), y = Math.ceil(num / x);
    const code = `@group(0) @binding(0) var<storage,read> mask:array<f32>;
@group(0) @binding(1) var<storage,read_write> outIdx:array<f32>;
@group(0) @binding(2) var<storage,read_write> outLogw:array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 let row=gid.x+gid.y*${x * 64}u;if(row>=${rows}u){return;}
 var slot=0u;
 for(var j=0u;j<${rows}u;j++){
  let m=mask[row*${rows}u+j];
  if(m!=0.0&&slot<${k}u){
   outIdx[row*${k}u+slot]=f32(j);
   outLogw[row*${k}u+slot]=${clampOne ? '0.0' : 'log(m)'};
   slot++;
  }
 }
 for(;slot<${k}u;slot++){
  outIdx[row*${k}u+slot]= -1.0;
  outLogw[row*${k}u+slot]= -1e30;
 }
}`;
    this.dispatch(code, [mask, idx, logw], [x, y], 'RFD3 mask to KNN');
    return { idx, logw, k };
  }
  gatherPairs(pair, knnIdx, rows, k, channels) {
    return this.kernel([pair, knnIdx], rows * k * channels, `
 let slotIdx=idx/${channels}u;let c=idx%${channels}u;let row=slotIdx/${k}u;
 let col=i32(a1[slotIdx]);
 if(col<0){out[idx]=0.0;return;}
 out[idx]=a0[(row*${rows}u+u32(col))*${channels}u+c];`, 'RFD3 gather KNN pairs');
  }
  attention(q, k, v, { queries, keys, channels, heads, bias, mask, knn, gate, qPreScale = false }) {
    return this.scope(() => {
      const dim = channels / heads;
      if (knn) {
        const width = knn.k, inputs = [q, k, knn.idx];
        let biasIndex, logwIndex;
        if (bias) { biasIndex = inputs.length; inputs.push(bias); }
        if (knn.logw) { logwIndex = inputs.length; inputs.push(knn.logw); }
        const scores = this.kernel(inputs, queries * heads * width, `let slot=idx%${width}u;let head=(idx/${width}u)%${heads}u;let query=idx/${width * heads}u;
 let colSigned=i32(a2[query*${width}u+slot]);
 if(colSigned<0){out[idx]= -1e30;return;}
 let key=u32(colSigned);
 var sum=0.0;for(var c=0u;c<${dim}u;c++){sum+=${qPreScale ? `(a0[query*${channels}u+head*${dim}u+c]/sqrt(${dim}.0))` : `a0[query*${channels}u+head*${dim}u+c]`}*a1[key*${channels}u+head*${dim}u+c];}
 out[idx]=${qPreScale ? 'sum' : `sum/sqrt(${dim}.0)`}${bias ? `+a${biasIndex}[(query*${width}u+slot)*${heads}u+head]` : ''}${knn.logw ? `+a${logwIndex}[query*${width}u+slot]` : ''};`, 'RFD3 sparse attention scores');
        const prob = this.softmax(scores, width);
        const values = [prob, v, knn.idx];
        let gateIndex;
        if (gate) { gateIndex = values.length; values.push(gate); }
        return this.kernel(values, queries * channels, `let query=idx/${channels}u;let head=(idx%${channels}u)/${dim}u;let c=idx%${dim}u;var sum=0.0;
 for(var slot=0u;slot<${width}u;slot++){let colSigned=i32(a2[query*${width}u+slot]);if(colSigned>=0){let p=a0[(query*${heads}u+head)*${width}u+slot];if(p!=0.0){sum+=p*a1[u32(colSigned)*${channels}u+head*${dim}u+c];}}}
 out[idx]=sum${gate ? `*a${gateIndex}[idx]` : ''};`, 'RFD3 sparse attention values');
      }
      const inputs = [q, k];
      let biasIndex, maskIndex;
      if (bias) { biasIndex = inputs.length; inputs.push(bias); }
      if (mask) { maskIndex = inputs.length; inputs.push(mask); }
      const scores = this.kernel(inputs, queries * heads * keys, `let key=idx%${keys}u;let head=(idx/${keys}u)%${heads}u;let query=idx/${keys * heads}u;
${mask ? `if(a${maskIndex}[query*${keys}u+key]==0.0){out[idx]=-1e30;return;}` : ''}
 var sum=0.0;for(var c=0u;c<${dim}u;c++){sum+=${qPreScale ? `(a0[query*${channels}u+head*${dim}u+c]/sqrt(${dim}.0))` : `a0[query*${channels}u+head*${dim}u+c]`}*a1[key*${channels}u+head*${dim}u+c];}
 out[idx]=${qPreScale ? 'sum' : `sum/sqrt(${dim}.0)`}${bias ? `+a${biasIndex}[(query*${keys}u+key)*${heads}u+head]` : ''}${mask ? `+log(a${maskIndex}[query*${keys}u+key])` : ''};`, 'RFD3 attention scores');
      const prob = this.softmax(scores, keys);
      const values = [prob, v];
      if (gate) values.push(gate);
      return this.kernel(values, queries * channels, `let query=idx/${channels}u;let head=(idx%${channels}u)/${dim}u;let c=idx%${dim}u;var sum=0.0;
 for(var key=0u;key<${keys}u;key++){let p=a0[(query*${heads}u+head)*${keys}u+key];if(p!=0.0){sum+=p*a1[key*${channels}u+head*${dim}u+c];}}out[idx]=sum${gate ? '*a2[idx]' : ''};`, 'RFD3 attention values');
    });
  }
  sortRanks(a, width) {
    const size = 2 ** Math.ceil(Math.log2(width)), rows = a.length / width, out = this.tensor(a.length);
    const code = `@group(0) @binding(0) var<storage,read> a:array<f32>;@group(0) @binding(1) var<storage,read_write> out:array<f32>;
var<workgroup> value:array<f32,${size}>;var<workgroup> index:array<u32,${size}>;
@compute @workgroup_size(${size}) fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) id:u32){value[id]=3.402823e38;index[id]=id;workgroupBarrier();if(id<${width}u){var maximum=-3.4e38;for(var j=0u;j<${width}u;j++){maximum=max(maximum,a[g.x*${width}u+j]);}let v=a[g.x*${width}u+id];var before=0u;var lower=0u;for(var j=0u;j<${width}u;j++){let other=a[g.x*${width}u+j];if(other<maximum){lower++;if(j<id&&v<maximum){before++;}}else if(j<id&&v==maximum){before++;}}let position=before+select(0u,lower,v==maximum);value[position]=v;index[position]=id;}workgroupBarrier();
for(var span=2u;span<=${size}u;span*=2u){for(var stride=span/2u;stride>0u;stride/=2u){let other=id^stride;if(other>id){let descending=(id&span)!=0u;if((value[id]<value[other])==descending){let v=value[id];value[id]=value[other];value[other]=v;let i=index[id];index[id]=index[other];index[other]=i;}}workgroupBarrier();}}
if(index[id]<${width}u){out[g.x*${width}u+index[id]]=f32(id);}}`;
    this.dispatch(code, [a, out], [rows], 'RFD3 small neighbour rank');
    return out;
  }
  softmax(a, channels) {
    const rows = a.length / channels, out = this.tensor(a.length);
    const code = `@group(0) @binding(0) var<storage,read> a:array<f32>;@group(0) @binding(1) var<storage,read_write> out:array<f32>;
var<workgroup> tmp:array<f32,64>;var<workgroup> maximum:f32;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) g:vec3<u32>,@builtin(local_invocation_index) id:u32){let row=g.x+g.y*65535u;if(row>=${rows}u){return;}var m=-1e30;
for(var c=id;c<${channels}u;c+=64u){m=max(m,a[row*${channels}u+c]);}tmp[id]=m;workgroupBarrier();
for(var s=32u;s>0u;s/=2u){if(id<s){tmp[id]=max(tmp[id],tmp[id+s]);}workgroupBarrier();}if(id==0u){maximum=tmp[0];}workgroupBarrier();
var sum=0.0;for(var c=id;c<${channels}u;c+=64u){let p=exp(a[row*${channels}u+c]-maximum);out[row*${channels}u+c]=p;sum+=p;}tmp[id]=sum;workgroupBarrier();
for(var s=32u;s>0u;s/=2u){if(id<s){tmp[id]+=tmp[id+s];}workgroupBarrier();}for(var c=id;c<${channels}u;c+=64u){out[row*${channels}u+c]/=tmp[0];}}`;
    this.dispatch(code, [a, out], [Math.min(rows, 65535), Math.ceil(rows / 65535)], 'RFD3 softmax');
    return out;
  }
  async submit() {
    if (!this.encoder) {
      this.drainDeferredReleases();
      return;
    }
    this.endPass();
    const encoder = this.encoder;
    this.encoder = null;
    this.device.queue.submit([encoder.finish()]);
    this.drainDeferredReleases();
    await this.device.queue.onSubmittedWorkDone();
  }
  async read(t) {
    const [data] = await this.readMany([t]);
    return data;
  }
  async readMany(tensors) {
    if (!tensors.length) return [];
    let totalBytes = 0;
    const offsets = [];
    for (const t of tensors) {
      offsets.push(totalBytes);
      totalBytes += t.length * 4;
    }
    const buffer = this.owner.allocate(totalBytes, 1 | 8, 'RFD3 readback');
    try {
      const enc = this.device.createCommandEncoder();
      for (let i = 0; i < tensors.length; i++) {
        enc.copyBufferToBuffer(tensors[i].buffer, 0, buffer, offsets[i], tensors[i].length * 4);
      }
      this.device.queue.submit([enc.finish()]);
      await buffer.mapAsync(1);
      const mapped = buffer.getMappedRange();
      const out = tensors.map((t, i) => new Float32Array(mapped.slice(offsets[i], offsets[i] + t.length * 4)));
      buffer.unmap();
      return out;
    } catch (err) {
      if (this.owner.closed && this.owner.loss) {
        throw new Error('GPU resources destroyed or device lost' + formatGpuLossDetail(this.owner.loss));
      }
      throw err;
    } finally {
      if (!this.owner.closed) this.owner.release(buffer);
    }
  }
  trim() {
    if (this.encoder) throw new Error('Cannot trim pending GPU commands');
    this.drainDeferredReleases();
    for (const bucket of this.pool.values()) for (const b of bucket) this.owner.release(b);
    this.pool.clear();
    this.poolBytes = 0;
  }
  destroy() {
    try { this.endPass(); } catch { /* ignore */ }
    this.encoder = null;
    this.pass = null;
    this.drainDeferredReleases();
    for (const t of [...this.tensors]) {
      this.owner.release(t.buffer);
      t.released = true;
    }
    this.tensors.clear();
    for (const bucket of this.pool.values()) for (const b of bucket) this.owner.release(b);
    this.pool.clear();
    this.poolBytes = 0;
    this.pipelines.clear();
  }
}
