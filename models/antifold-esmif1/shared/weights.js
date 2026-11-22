/** Safetensors storage and integrity. Browser, worker, and injected-reader safe. */
export const sizeOf = { F32: 4, F16: 2, I8: 1, I32: 4 };
export function halfToFloat(h) {
  const s = h & 32768 ? -1 : 1, e = h >>> 10 & 31, m = h & 1023;
  return s * (e === 0 ? m * 2 ** -24 : e === 31 ? (m ? NaN : Infinity) : (1 + m / 1024) * 2 ** (e - 15));
}
export async function sha256(data) {
  if (!globalThis.crypto?.subtle) throw new Error('Web Crypto is required for SHA-256 integrity checks');
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', data)), b => b.toString(16).padStart(2, '0')).join('');
}
export function parseSafetensors(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 10) throw new Error('Invalid safetensors');
  const n = Number(new DataView(buffer).getBigUint64(0, true));
  if (!Number.isSafeInteger(n) || n < 2 || n + 8 > buffer.byteLength) throw new Error('Invalid safetensors header');
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, n))), out = Object.create(null), ranges = [];
  for (const [name, t] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const [a, b] = t.data_offsets, width = sizeOf[t.dtype];
    if (!width || !t.shape.every(d => Number.isSafeInteger(d) && d > 0) || !Number.isSafeInteger(a) || !Number.isSafeInteger(b) || a < 0 || b < a || n + 8 + b > buffer.byteLength || b - a !== t.shape.reduce((a, b) => a * b, 1) * width) throw new Error(`Invalid tensor ${name}`);
    ranges.push([a,b]);out[name] = { dtype: t.dtype, shape: t.shape, bytes: new Uint8Array(buffer, n + 8 + a, b - a) };
  }
  ranges.sort((a,b)=>a[0]-b[0]);for(let i=1;i<ranges.length;i++) if(ranges[i][0]<ranges[i-1][1]) throw new Error('Overlapping tensors');
  return out;
}
export function decodeTensor(t) {
  const n = t.shape.reduce((a,b)=>a*b,1), v = new DataView(t.bytes.buffer,t.bytes.byteOffset,t.bytes.byteLength), out = new Float32Array(n);
  for(let i=0;i<n;i++) out[i]=t.dtype==='F32'?v.getFloat32(4*i,true):t.dtype==='F16'?halfToFloat(v.getUint16(2*i,true)):v.getInt8(i)*t.scales[Math.floor(i/t.shape[1])];
  return out;
}
export function createReader({read,cache='none',signal,onProgress}={}) {
  if(!['none','cache','opfs'].includes(cache)) throw new Error('Unknown cache mode');
  return async url => {
    signal?.throwIfAborted();
    if(read) {const data=await read(url);signal?.throwIfAborted();return data;}
    let storage,key;
    try {
      if(cache==='cache' && globalThis.caches) {storage=await caches.open('gvp-transformer-v1');const hit=await storage.match(url);if(hit)return await hit.arrayBuffer();}
      if(cache==='opfs' && globalThis.navigator?.storage?.getDirectory) {
        storage=await (await navigator.storage.getDirectory()).getDirectoryHandle('gvp-transformer-v1',{create:true});
        key=await sha256(new TextEncoder().encode(new URL(url,globalThis.location?.href).href));
        try {return await (await (await storage.getFileHandle(key)).getFile()).arrayBuffer();} catch(e) {if(e.name!=='NotFoundError')throw e;}
      }
    } catch(e) {storage=null;onProgress?.({phase:'cache-warning',message:e.message});}
    const response=await fetch(url,{signal});if(!response.ok)throw new Error(`HTTP ${response.status}: ${url}`);
    const data=await response.arrayBuffer();signal?.throwIfAborted();
    if(storage)try {
      if(key){const writer=await (await storage.getFileHandle(key,{create:true})).createWritable();try{await writer.write(data);await writer.close();}catch(e){await writer.abort().catch(()=>{});throw e;}}
      else await storage.put(url,new Response(data));
    }catch(e){onProgress?.({phase:'cache-warning',message:e.message});}
    return data;
  };
}
export async function loadWeights(baseUrl,options={}) {
  const read=createReader(options),base=String(baseUrl).replace(/\/$/,''),manifest=JSON.parse(new TextDecoder().decode(await read(`${base}/manifest.json`)));
  if(manifest.format!=='gvp-transformer-safetensors-v1'||!['antifold','esmif1'].includes(manifest.model)||!['fp32','fp16','int8'].includes(manifest.precision)) throw new Error('Unsupported model manifest');
  const byShard=new Map(),pending=new Map(),weights=Object.create(null);
  if(manifest.num_tensors!==Object.keys(manifest.tensors).length)throw new Error('Tensor count mismatch');
  const names=new Set(manifest.shards.map(s=>s.file));
  for(const [name,t] of Object.entries(manifest.tensors)) {
    if(!sizeOf[t.dtype]||!Array.isArray(t.shape)||!t.shape.every(n=>Number.isSafeInteger(n)&&n>0))throw new Error(`Invalid tensor ${name}`);
    const parts=t.parts??[t];let row=0;
    for(const p of parts){
      if(!names.has(p.shard)||p.dtype!==t.dtype||(p.row_start??0)!==row||p.shape.slice(1).toString()!==t.shape.slice(1).toString())throw new Error(`Invalid multipart tensor ${name}`);
      row+=p.shape[0];if(!byShard.has(p.shard))byShard.set(p.shard,[]);byShard.get(p.shard).push({name,t,p});
    }
    if(row!==t.shape[0])throw new Error(`Incomplete rows: ${name}`);
  }
  let loaded=0;
  for(const shard of manifest.shards){
    options.signal?.throwIfAborted();
    if(!/^[\w.-]+\.safetensors$/.test(shard.file)||shard.byteLength>25_000_000)throw new Error('Invalid shard');
    const data=await read(`${base}/${shard.file}`);
    if(data.byteLength!==shard.byteLength)throw new Error(`Shard byte length mismatch: ${shard.file}`);
    if(await sha256(data)!==shard.sha256)throw new Error(`Shard SHA-256 mismatch: ${shard.file}`);
    const parsed=parseSafetensors(data);
    for(const {name,t,p} of byShard.get(shard.file)??[]){
      const raw=parsed[p.key??name];if(!raw||raw.dtype!==p.dtype||raw.shape.toString()!==p.shape.toString())throw new Error(`Manifest mismatch: ${name}`);
      let out=pending.get(name);
      if(!out){out={shape:t.shape,dtype:t.dtype,bytes:new Uint8Array(t.shape.reduce((a,b)=>a*b,1)*sizeOf[t.dtype]),remaining:(t.parts??[t]).length};if(t.dtype==='I8')out.scales=new Float32Array(t.shape[0]);pending.set(name,out);}
      const stride=t.shape.slice(1).reduce((a,b)=>a*b,1);out.bytes.set(raw.bytes,(p.row_start??0)*stride*sizeOf[t.dtype]);
      if(t.dtype==='I8'){
        const s=parsed[p.quant?.scale_tensor];if(p.quant?.scheme!=='symmetric_per_row'||!s||s.dtype!=='F32'||s.shape.toString()!==String(p.shape[0]))throw new Error(`Missing colocated scales: ${name}`);
        const scales=decodeTensor(s);if(scales.some(x=>!Number.isFinite(x)||x<=0))throw new Error('Invalid quantization scale');out.scales.set(scales,p.row_start??0);
      }
      if(--out.remaining===0){delete out.remaining;pending.delete(name);if(options.onTensor)await options.onTensor(name,out);else weights[name]=out;}
    }
    loaded+=data.byteLength;options.onProgress?.({phase:'weights',loaded,total:manifest.total_bytes});
  }
  if(pending.size||loaded!==manifest.total_bytes)throw new Error('Incomplete bundle');
  options.signal?.throwIfAborted();
  return {manifest,weights};
}
export function diffStats(a,b) {
  if(a.length!==b.length)throw new Error(`Shape mismatch ${a.length} != ${b.length}`);
  let diff=0,ref=0,maxAbs=0;
  for(let i=0;i<a.length;i++){if(!Number.isFinite(a[i])||!Number.isFinite(b[i]))throw new Error(`Nonfinite value at ${i}`);const d=a[i]-b[i];diff+=d*d;ref+=b[i]*b[i];maxAbs=Math.max(maxAbs,Math.abs(d));}
  return {relRMS:Math.sqrt(diff/Math.max(ref,1e-30)),maxAbs};
}
