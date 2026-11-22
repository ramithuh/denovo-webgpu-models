/** Browser/worker-safe sequence input and binary bundle utilities. No dependencies. */
export const VOCAB = ['<cls>','<pad>','<eos>','<unk>',...'LAGVSERTIDPKQNFYMHWCXBUZO','.', '-', '|', '<mask>'];
export function tokenize(sequence, maxLength = 1024) {
  if (typeof sequence !== 'string') throw new TypeError('Expected an amino acid sequence');
  sequence = sequence.replace(/\s/g, '').toUpperCase();
  if (!sequence.length || sequence.length > maxLength || !/^[ACDEFGHIKLMNPQRSTVWY]+$/.test(sequence))
    throw new RangeError(`Expected 1–${maxLength} canonical amino acids`);
  return { residues: sequence, tokens: Int32Array.from(sequence, c => VOCAB.indexOf(c)) };
}
export function halfToFloat(h) {
  const s = h & 32768 ? -1 : 1, e = (h >>> 10) & 31, m = h & 1023;
  return s * (e === 0 ? m * 2 ** -24 : e === 31 ? (m ? NaN : Infinity) : (1 + m / 1024) * 2 ** (e - 15));
}
export function parseSafetensors(buffer) {
  const view = new DataView(buffer);
  const n = Number(view.getBigUint64(0, true));
  if (!Number.isSafeInteger(n) || n < 2 || n + 8 > buffer.byteLength) throw new Error('Invalid safetensors header');
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, n)));
  const tensors = {};
  for (const [name, rec] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const [a,b] = rec.data_offsets, bytesPerElement = {F32:4,F16:2,I8:1,I32:4}[rec.dtype];
    if (!bytesPerElement || a < 0 || b < a || 8+n+b > buffer.byteLength || b-a !== rec.shape.reduce((a,b)=>a*b,1)*bytesPerElement)
      throw new Error(`Invalid tensor: ${name}`);
    tensors[name] = { dtype: rec.dtype, shape: rec.shape, bytes: new Uint8Array(buffer,8+n+a,b-a) };
  }
  return tensors;
}
export function decodeTensor(t) {
  const count = t.shape.reduce((a,b)=>a*b,1), out = new Float32Array(count);
  const v = new DataView(t.bytes.buffer,t.bytes.byteOffset,t.bytes.byteLength);
  const width = t.shape[1] ?? 1;
  for (let i=0;i<count;i++) out[i] = t.dtype === 'F32' ? v.getFloat32(i*4,true) : t.dtype === 'F16' ? halfToFloat(v.getUint16(i*2,true)) : v.getInt8(i)*t.scales[Math.floor(i/width)];
  return out;
}
export function diffStats(a,b) {
  if(a.length!==b.length) throw new Error('Shape mismatch');
  let d=0,r=0,maxAbs=0,sa=0,sb=0,aa=0,bb=0,ab=0;
  for(let i=0;i<a.length;i++) {if(!Number.isFinite(a[i])) throw new Error(`Nonfinite value at ${i}`);const e=a[i]-b[i];d+=e*e;r+=b[i]*b[i];maxAbs=Math.max(maxAbs,Math.abs(e));sa+=a[i];sb+=b[i];aa+=a[i]*a[i];bb+=b[i]*b[i];ab+=a[i]*b[i];}
  return {relRms:Math.sqrt(d/Math.max(r,1e-30)),maxAbs,correlation:(ab-sa*sb/a.length)/Math.sqrt(Math.max(1e-30,(aa-sa*sa/a.length)*(bb-sb*sb/a.length)))};
}
export async function loadWeights(baseUrl, { read, signal, onProgress, cache = 'none', onTensor } = {}) {
  const fetchBytes = async url => {
    signal?.throwIfAborted();
    if (read) return read(url);
    let storage, key, cached;
    if (cache === 'cache' && globalThis.caches) { storage = await caches.open('dyna1-v1'); cached = await storage.match(url); }
    if (cache === 'opfs' && globalThis.navigator?.storage?.getDirectory) {
      storage = await (await navigator.storage.getDirectory()).getDirectoryHandle('dyna1-v1',{create:true});
      key = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(url))),x=>x.toString(16).padStart(2,'0')).join('');
      try { return await (await (await storage.getFileHandle(key)).getFile()).arrayBuffer(); } catch(e) { if(e.name !== 'NotFoundError') throw e; }
    }
    if (cached) return cached.arrayBuffer();
    const response = await fetch(url,{signal});
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${url}`);
    const data = await response.arrayBuffer();
    if (storage) {
      try {
        if (key) {const writer=await (await storage.getFileHandle(key,{create:true})).createWritable();await writer.write(data);await writer.close();}
        else await storage.put(url,new Response(data));
      } catch(e) { onProgress?.({phase:'cache-warning',message:e.message}); }
    }
    return data;
  };
  const base = baseUrl.replace(/\/$/,'');
  const manifest = JSON.parse(new TextDecoder().decode(await fetchBytes(`${base}/manifest.json`)));
  if(manifest.format !== 'dyna1-safetensors-v1') throw new Error('Unsupported Dyna-1 bundle format');
  const byShard = new Map(), pending = new Map(), weights = {};
  for (const [name,rec] of Object.entries(manifest.tensors)) {
    for(const part of rec.parts ?? [rec]) {
      if(!byShard.has(part.shard)) byShard.set(part.shard,[]);
      byShard.get(part.shard).push({name,rec,part});
    }
  }
  let loaded=0;
  for(const shard of manifest.shards) {
    signal?.throwIfAborted();
    const data = await fetchBytes(`${base}/${shard.file}`);
    if(data.byteLength !== shard.bytes) throw new Error(`Shard size mismatch: ${shard.file}`);
    if(globalThis.crypto?.subtle) {
      const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',data)),x=>x.toString(16).padStart(2,'0')).join('');
      if(hash!==shard.sha256) throw new Error(`Shard checksum mismatch: ${shard.file}`);
    }
    const parsed=parseSafetensors(data);
    for(const {name,rec,part} of byShard.get(shard.file) ?? []) {
      const raw=parsed[part.key ?? name];
      if(!raw || raw.dtype!==part.dtype || raw.shape.toString()!==part.shape.toString()) throw new Error(`Manifest mismatch: ${name}`);
      let t=pending.get(name);
      if(!t) {t={dtype:rec.dtype,shape:rec.shape,bytes:new Uint8Array(rec.shape.reduce((a,b)=>a*b,1)*({F32:4,F16:2,I8:1}[rec.dtype])),remaining:(rec.parts??[rec]).length};if(rec.dtype==='I8') t.scales=new Float32Array(rec.shape[0]);pending.set(name,t);}
      const row=part.row_start??0, stride=(rec.shape[1]??1)*({F32:4,F16:2,I8:1}[rec.dtype]);
      t.bytes.set(raw.bytes,row*stride);
      if(part.quant) t.scales.set(decodeTensor(parsed[part.quant.scale_tensor]),row);
      if(--t.remaining===0) {delete t.remaining;pending.delete(name);if(onTensor) await onTensor(name,t);else weights[name]=t;}
    }
    loaded+=shard.bytes;onProgress?.({phase:'weights',loaded,total:manifest.total_bytes,file:shard.file});
  }
  if(pending.size) throw new Error('Incomplete bundle');
  return {manifest,config:manifest.config,weights};
}
