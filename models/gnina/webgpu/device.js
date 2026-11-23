import {shaders} from "./shaders.js";
import {checkAbort} from "../shared/loader.js";

function fallback(info) {
  return info.isFallbackAdapter===true||/swiftshader|llvmpipe|lavapipe|software|\bwarp\b|basic render/i.test([info.vendor,info.architecture,info.description,info.device].join(" "));
}
export async function acquire({device,allowSoftwareFallback=true,signal}) {
  let adapter;
  if(!device) {
    const gpu=globalThis.navigator?.gpu;
    if(!gpu) throw Error("WebGPU unavailable");
    for(const options of [{powerPreference:"high-performance"},{powerPreference:"high-performance"},{}]) {
      checkAbort(signal);adapter=await gpu.requestAdapter(options);if(adapter) break;
    }
    if(!adapter) throw Error("WebGPU adapter unavailable");
  }
  const raw=adapter?.info??device?.adapterInfo??{};
  const adapterInfo=Object.fromEntries(["vendor","architecture","device","description","isFallbackAdapter"].map(k=>[k,raw[k]]));
  const isFallbackAdapter=fallback(adapterInfo)||adapter?.isFallbackAdapter===true;
  if(!allowSoftwareFallback && (isFallbackAdapter||(!adapterInfo.vendor&&!adapterInfo.description&&!adapterInfo.architecture))) throw Error("Verified hardware WebGPU adapter required");
  const owns=!device;
  device??=await adapter.requestDevice();
  return {device,owns,adapterInfo,isFallbackAdapter};
}
export class Resources {
  constructor(connection) {
    Object.assign(this,connection);
    this.buffers=new Map();this.residentBytes=0;this.transientBytes=0;this.peakBytes=0;this.pipelines={};
    this.device.lost.then(reason=>{this.lost=reason.message||reason.reason;});
  }
  alive() {
    if(this.dead||this.lost) throw Error(this.lost??"WebGPU engine destroyed");
  }
  allocate(size,usage,resident=false,data) {
    this.alive();
    size=Math.max(4,Math.ceil(size/4)*4);
    if(!Number.isSafeInteger(size)||size>this.device.limits.maxBufferSize||
      ((usage&GPUBufferUsage.STORAGE)&&size>this.device.limits.maxStorageBufferBindingSize)) throw Error("GPU allocation exceeds device limits");
    const b=this.device.createBuffer({size,usage,mappedAtCreation:!!data});
    this.buffers.set(b,resident);this[resident?"residentBytes":"transientBytes"]+=size;
    this.peakBytes=Math.max(this.peakBytes,this.residentBytes+this.transientBytes);
    if(data) {
      new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength));
      b.unmap();
    }
    return b;
  }
  free(b) {
    if(!this.buffers.has(b)) return;
    this[this.buffers.get(b)?"residentBytes":"transientBytes"]-=b.size;
    this.buffers.delete(b);b.destroy();
  }
  memoryUsage() {
    const {residentBytes,transientBytes,peakBytes}=this;
    return {residentBytes,transientBytes,peakBytes,totalBytes:residentBytes+transientBytes,buffers:this.buffers.size};
  }
  async initialize(tensors,names,signal) {
    const chunks=[], align=n=>Math.ceil(n/4)*4;let size=0;
    function append(bytes) {const offset=size;chunks.push([offset,bytes]);size+=align(bytes.byteLength);return offset/4;}
    this.layout={};
    for(const [key,t] of Object.entries(tensors)) {
      if(!names.includes(key.split("/")[0])) continue;
      const offset=append(t.bytes), scale=t.scale?append(new Uint8Array(t.scale.buffer)):0;
      this.layout[key]={shape:t.shape,offset,scale,kind:{F32:0,F16:1,I8:2}[t.dtype]};
    }
    const bytes=new Uint8Array(size);
    for(const [offset,chunk] of chunks) bytes.set(chunk,offset);
    this.packed=this.allocate(size,GPUBufferUsage.STORAGE,true,bytes);
    await Promise.all(Object.entries(shaders).map(async ([label,code])=>{
      const module=this.device.createShaderModule({label,code});
      this.pipelines[label]=await this.device.createComputePipelineAsync({label,layout:"auto",compute:{module,entryPoint:"main"}});
    }));
    checkAbort(signal);this.alive();
  }
  destroy() {
    if(this.dead) return;
    this.dead=true;for(const b of this.buffers.keys()) this.free(b);
    this.pipelines={};if(this.owns) this.device.destroy();
  }
}
