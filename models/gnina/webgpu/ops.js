const block=256;
const PARAM_SLOTS=256;

function fold2(groups) {
  return groups<=65535?[Math.max(1,groups),1]:[65535,Math.ceil(groups/65535)];
}

export class GPUOps {
  constructor(resources,check) {
    this.gpu=resources;this.check=check;this.owned=new Set();this.poolBySize=new Map();
    this.device=resources.device;this.encoder=this.device.createCommandEncoder();
    this.pass=null;
    this.paramData=new Uint32Array(PARAM_SLOTS*64);
    this.paramBuffer=this.buffer(this.paramData.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
    this.paramCount=0;
  }
  buffer(size,usage,data) {
    const b=this.gpu.allocate(size,usage,false,data);this.owned.add(b);return b;
  }
  tensor(c,d) {
    const size=c*d**3*4;
    const b=this.poolBySize.get(size)?.pop()??this.buffer(size,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST);
    return {c,d,b,size};
  }
  release({b,size}) {
    if(!this.poolBySize.has(size)) this.poolBySize.set(size,[]);
    this.poolBySize.get(size).push(b);
  }
  _endPass() {
    if(this.pass){this.pass.end();this.pass=null;}
  }
  _flushParams() {
    if(this.paramCount>0) {
      this.device.queue.writeBuffer(this.paramBuffer,0,this.paramData,0,this.paramCount*64);
    }
  }
  dispatch(name,bindings,params,x,y=1) {
    this.check();this.gpu.alive();
    if(y===1 && x>65535) [x,y]=fold2(x);
    if(Math.max(x,y)>this.device.limits.maxComputeWorkgroupsPerDimension) throw Error("Dispatch exceeds device limit");
    const entries=[];
    for(const [index,buffer] of Object.entries(bindings)) {
      entries.push({binding:Number(index),resource:{buffer}});
    }
    if(params) {
      if(this.paramCount>=PARAM_SLOTS) throw Error("Parameter slot overflow");
      const slot=this.paramCount++;
      const words=params.buffer.byteLength===params.byteLength && params instanceof Uint32Array
        ? params
        : new Uint32Array(params.buffer,params.byteOffset,params.byteLength/4);
      this.paramData.set(words,slot*64);
      entries.push({binding:3,resource:{buffer:this.paramBuffer,offset:slot*256,size:Math.max(16,words.byteLength)}});
    }
    const pipeline=this.gpu.pipelines[name];
    const group=this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries});
    if(!this.pass) this.pass=this.encoder.beginComputePass();
    this.pass.setPipeline(pipeline);this.pass.setBindGroup(0,group);this.pass.dispatchWorkgroups(x,y);
  }
  async read(tensor) {
    const [arr]=await this.readMany([tensor]);
    return arr;
  }
  async readMany(tensors) {
    this.check();this.gpu.alive();
    this._endPass();
    this._flushParams();
    const sizes=tensors.map(t=>Math.max(4,t.size));
    const total=sizes.reduce((a,b)=>a+b,0);
    const staging=this.buffer(total,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);
    try {
      let off=0;
      for(let i=0;i<tensors.length;i++) {
        this.encoder.copyBufferToBuffer(tensors[i].b,0,staging,off,tensors[i].size);
        off+=sizes[i];
      }
      this.device.queue.submit([this.encoder.finish()]);
      this.encoder=this.device.createCommandEncoder();
      this.paramCount=0;
      await staging.mapAsync(GPUMapMode.READ);
      this.check();this.gpu.alive();
      const mapped=staging.getMappedRange();
      const out=[];
      let pos=0;
      for(let i=0;i<tensors.length;i++) {
        out.push(new Float32Array(mapped.slice(pos,pos+tensors[i].size)));
        pos+=sizes[i];
      }
      return out;
    } finally {this.gpu.free(staging);this.owned.delete(staging);}
  }
  finish() {this._endPass();for(const b of this.owned) this.gpu.free(b);this.owned.clear();this.poolBySize.clear();this.encoder=null;}
  grid({origin,groups}) {
    const out=this.tensor(28,48),starts=new Uint32Array(29);
    for(let i=0;i<28;i++) starts[i+1]=starts[i]+groups[i].length;
    const atoms=new Float32Array(Math.max(1,starts[28])*4);let i=0;
    for(const group of groups) for(const a of group) {atoms.set([a.x,a.y,a.z,a.radius],i);i+=4;}
    this.dispatch("grid",{0:this.buffer(atoms.byteLength,GPUBufferUsage.STORAGE,atoms),1:this.buffer(starts.byteLength,GPUBufferUsage.STORAGE,starts),2:out.b},Float32Array.from(origin),Math.ceil(out.size/4/block));
    return out;
  }
  pool(x,kind,k) {
    const out=this.tensor(x.c,x.d/k);
    this.dispatch("pool",{0:x.b,2:out.b},Uint32Array.of(x.d,k,Number(kind==="avg"),out.size/4),Math.ceil(out.size/4/block));
    return out;
  }
  conv(x,name,negativeSlope=0) {
    const w=this.gpu.layout[name+".weight"],bias=this.gpu.layout[name+".bias"],[oc,,k]=w.shape;
    const out=this.tensor(oc,x.d);
    this.dispatch("conv",{0:x.b,1:this.gpu.packed,2:out.b},Uint32Array.of(x.d,x.c,oc,k,w.offset,w.scale,w.kind,bias.offset,new Uint32Array(Float32Array.of(negativeSlope).buffer)[0]),Math.ceil(x.d**3/128),Math.ceil(oc/16));
    return out;
  }
  norm(x,name) {
    const offsets=["weight","bias","running_mean","running_var"].map(k=>this.gpu.layout[name+"."+k].offset);
    const out=this.tensor(x.c,x.d);
    this.dispatch("norm",{0:x.b,1:this.gpu.packed,2:out.b},Uint32Array.of(x.d**3,x.size/4,...offsets),Math.ceil(x.size/4/block));
    return out;
  }
  concat(x,y) {
    const out=this.tensor(x.c+y.c,x.d);
    this.dispatch("concat",{0:x.b,1:y.b,2:out.b},Uint32Array.of(x.size/4,out.size/4),Math.ceil(out.size/4/block));
    return out;
  }
  heads(x,pose,affinity) {
    const logits=this.tensor(3,1),out=this.tensor(2,1),params=[x.size/4];
    for(const name of [pose,affinity]) {
      const w=this.gpu.layout[name+".weight"];
      params.push(w.offset,w.scale,w.kind,this.gpu.layout[name+".bias"].offset);
    }
    this.dispatch("linear",{0:x.b,1:this.gpu.packed,2:logits.b},Uint32Array.from(params),3);
    this.dispatch("softmax",{0:logits.b,2:out.b},null,1);this.release(logits);
    return out;
  }
}
