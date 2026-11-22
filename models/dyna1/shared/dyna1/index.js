/** Browser/worker-safe sequence, 3D backbone geometry, and binary bundle utilities. No dependencies. */
export const VOCAB = ['<cls>','<pad>','<eos>','<unk>',...'LAGVSERTIDPKQNFYMHWCXBUZO','.', '-', '|', '<mask>'];
export const SEQUENCE_MASK_TOKEN = 32;
export const STRUCTURE_MASK_TOKEN = 4096;
export const STRUCTURE_VOCAB_SIZE = 4101;

export function tokenize(sequence, maxLength = 1024) {
  if (typeof sequence !== 'string') throw new TypeError('Expected an amino acid sequence');
  sequence = sequence.replace(/\s/g, '').toUpperCase();
  if (!sequence.length || sequence.length > maxLength || !/^[ACDEFGHIKLMNPQRSTVWY]+$/.test(sequence))
    throw new RangeError(`Expected 1–${maxLength} canonical amino acids`);
  return { residues: sequence, tokens: Int32Array.from(sequence, c => VOCAB.indexOf(c)) };
}

function sanitizeCoord(v) {
  return typeof v === 'number' && Number.isFinite(v) && v < 1e6 ? v : NaN;
}

/** Normalize (N, CA, C) backbone coordinates into flat Float32Array [L * 9] (Å). */
export function normalizeBackboneCoords(input, maxLength = 1024) {
  if (!input || typeof input !== 'object') throw new TypeError('Expected backbone coordinates');
  let coords, length;
  if ('n' in input && 'ca' in input && 'c' in input) {
    const { n, ca, c } = input;
    if (!n || !ca || !c || n.length !== ca.length || n.length !== c.length || n.length % 3 !== 0)
      throw new RangeError('Expected { n, ca, c } coordinate arrays of equal length 3*L');
    length = n.length / 3;
    coords = new Float32Array(length * 9);
    for (let i = 0; i < length; i++) {
      for (let d = 0; d < 3; d++) {
        coords[i * 9 + d] = sanitizeCoord(n[i * 3 + d]);
        coords[i * 9 + 3 + d] = sanitizeCoord(ca[i * 3 + d]);
        coords[i * 9 + 6 + d] = sanitizeCoord(c[i * 3 + d]);
      }
    }
  } else if (Array.isArray(input) && input.length > 0 && Array.isArray(input[0])) {
    length = input.length;
    coords = new Float32Array(length * 9);
    for (let i = 0; i < length; i++) {
      const res = input[i];
      if (!Array.isArray(res) || res.length < 3) throw new RangeError(`Expected [N, CA, C] coordinates at residue ${i}`);
      for (let a = 0; a < 3; a++) {
        const atom = res[a];
        for (let d = 0; d < 3; d++) coords[i * 9 + a * 3 + d] = atom ? sanitizeCoord(atom[d]) : NaN;
      }
    }
  } else if (typeof input.length === 'number') {
    if (input.length % 9 !== 0) throw new RangeError('Expected flat backbone coordinates of length 9*L (N, CA, C)');
    length = input.length / 9;
    coords = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) coords[i] = sanitizeCoord(input[i]);
  } else {
    throw new TypeError('Unsupported structureCoords format');
  }
  if (!length || length > maxLength) throw new RangeError(`Expected 1–${maxLength} residues in structureCoords`);
  return { coords, length };
}

/** Normalize discrete ESM-3 structure tokens (0..4100; -1 maps to 4096 mask). */
export function normalizeStructureTokens(tokens, maxLength = 1024) {
  if (!tokens || typeof tokens.length !== 'number') throw new TypeError('Expected structureTokens array');
  const length = tokens.length;
  if (!length || length > maxLength) throw new RangeError(`Expected 1–${maxLength} structure tokens`);
  const out = new Int32Array(length);
  for (let i = 0; i < length; i++) {
    const raw = tokens[i];
    if (typeof raw !== 'number' || !Number.isInteger(raw) || raw < -1 || raw >= STRUCTURE_VOCAB_SIZE)
      throw new RangeError(`Invalid structure token ${raw} at index ${i}`);
    out[i] = raw === -1 ? STRUCTURE_MASK_TOKEN : raw;
  }
  return out;
}

export function normalizeResidueIndex(residueIndex, length) {
  if (residueIndex == null) return Int32Array.from({ length }, (_, i) => i);
  if (typeof residueIndex.length !== 'number' || residueIndex.length !== length)
    throw new RangeError(`Expected residueIndex of length ${length}`);
  const out = new Int32Array(length);
  for (let i = 0; i < length; i++) {
    const v = residueIndex[i];
    if (typeof v !== 'number' || !Number.isInteger(v)) throw new RangeError(`Invalid residueIndex at ${i}`);
    out[i] = v;
  }
  return out;
}

/** Resolve seq-only, seq+struct, or struct-only prediction inputs. */
export function resolvePredictInputs(sequence, opts = {}, maxLength = 1024) {
  const rawCoords = opts.structureCoords ?? opts.backbone3Coords ?? null;
  const rawStructTokens = opts.structureTokens ?? null;
  const hasStruct = rawCoords != null || rawStructTokens != null;
  const hasSeq = typeof sequence === 'string' && sequence.replace(/\s/g, '').length > 0;
  let mode = opts.mode;
  if (mode != null) {
    if (mode !== 'seq-only' && mode !== 'seq+struct' && mode !== 'struct-only')
      throw new RangeError(`Unknown mode: ${mode}`);
  } else {
    mode = !hasStruct ? 'seq-only' : hasSeq ? 'seq+struct' : 'struct-only';
  }
  if (mode === 'seq-only') {
    const { residues, tokens } = tokenize(sequence, maxLength);
    return { mode, length: tokens.length, residues, seqTokens: tokens, coords: null, structureTokens: null, residueIndex: null };
  }
  if (!hasStruct) throw new TypeError(`Mode '${mode}' requires structureCoords or structureTokens`);
  const coordsInfo = rawCoords != null ? normalizeBackboneCoords(rawCoords, maxLength) : null;
  const structureTokens = rawStructTokens != null ? normalizeStructureTokens(rawStructTokens, maxLength) : null;
  if (coordsInfo && structureTokens && coordsInfo.length !== structureTokens.length)
    throw new RangeError('structureCoords and structureTokens lengths must match');
  const length = coordsInfo ? coordsInfo.length : structureTokens.length;
  const residueIndex = normalizeResidueIndex(opts.residueIndex, length);
  if (mode === 'seq+struct') {
    const { residues, tokens } = tokenize(sequence, maxLength);
    if (tokens.length !== length)
      throw new RangeError(`Sequence length (${tokens.length}) does not match structure length (${length})`);
    return { mode, length, residues, seqTokens: tokens, coords: coordsInfo?.coords ?? null, structureTokens, residueIndex };
  }
  let residues = null;
  if (hasSeq) {
    const tok = tokenize(sequence, maxLength);
    if (tok.tokens.length !== length)
      throw new RangeError(`Sequence length (${tok.tokens.length}) does not match structure length (${length})`);
    residues = tok.residues;
  }
  const seqTokens = new Int32Array(length);
  seqTokens.fill(SEQUENCE_MASK_TOKEN);
  return { mode, length, residues, seqTokens, coords: coordsInfo?.coords ?? null, structureTokens, residueIndex };
}

const f32 = Math.fround;
function dot3(ax, ay, az, bx, by, bz) { return f32(f32(f32(ax * bx) + f32(ay * by)) + f32(az * bz)); }
function gramSchmidt(x0, x1, x2, v0, v1, v2) {
  const nx = f32(Math.sqrt(f32(dot3(x0, x1, x2, x0, x1, x2) + 1e-10)));
  const e00 = f32(x0 / nx), e01 = f32(x1 / nx), e02 = f32(x2 / nx);
  const d = dot3(e00, e01, e02, v0, v1, v2);
  const u0 = f32(v0 - f32(e00 * d)), u1 = f32(v1 - f32(e01 * d)), u2 = f32(v2 - f32(e02 * d));
  const nu = f32(Math.sqrt(f32(dot3(u0, u1, u2, u0, u1, u2) + 1e-10)));
  const e10 = f32(u0 / nu), e11 = f32(u1 / nu), e12 = f32(u2 / nu);
  const e20 = f32(f32(e01 * e12) - f32(e02 * e11));
  const e21 = f32(f32(e02 * e10) - f32(e00 * e12));
  const e22 = f32(f32(e00 * e11) - f32(e01 * e10));
  return [e00, e10, e20, e01, e11, e21, e02, e12, e22];
}

/** Exact ESM-3 build_affine3d_from_coordinates: returns Float32Array [L * 13] (9 rot + 3 trans + 1 mask). */
export function buildStructureFrames(coords, L) {
  const out = new Float32Array(L * 13);
  const valid = new Uint8Array(L);
  const avg = new Float64Array(9);
  let count = 0;
  for (let i = 0; i < L; i++) {
    let ok = 1;
    const base = i * 9;
    for (let j = 0; j < 9; j++) {
      const v = coords[base + j];
      if (!Number.isFinite(v) || v >= 1e6) { ok = 0; break; }
    }
    valid[i] = ok;
    if (ok) {
      count++;
      for (let j = 0; j < 9; j++) avg[j] += coords[base + j];
    }
  }
  const avgF = new Float32Array(9);
  const denom = count + 1e-8;
  for (let j = 0; j < 9; j++) avgF[j] = f32(avg[j] / denom);
  const fallbackRot = count > 0
    ? gramSchmidt(f32(avgF[3] - avgF[6]), f32(avgF[4] - avgF[7]), f32(avgF[5] - avgF[8]),
                  f32(avgF[0] - avgF[3]), f32(avgF[1] - avgF[4]), f32(avgF[2] - avgF[5]))
    : [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (let i = 0; i < L; i++) {
    const off = i * 13;
    if (valid[i]) {
      const b = i * 9;
      const rot = gramSchmidt(
        f32(coords[b + 3] - coords[b + 6]), f32(coords[b + 4] - coords[b + 7]), f32(coords[b + 5] - coords[b + 8]),
        f32(coords[b + 0] - coords[b + 3]), f32(coords[b + 1] - coords[b + 4]), f32(coords[b + 2] - coords[b + 5])
      );
      for (let k = 0; k < 9; k++) out[off + k] = rot[k];
      out[off + 9] = coords[b + 3]; out[off + 10] = coords[b + 4]; out[off + 11] = coords[b + 5];
      out[off + 12] = 1;
    } else {
      for (let k = 0; k < 9; k++) out[off + k] = fallbackRot[k];
      out[off + 9] = avgF[3]; out[off + 10] = avgF[4]; out[off + 11] = avgF[5];
      out[off + 12] = 0;
    }
  }
  return out;
}

/** Exact ESM-3 knn_graph on CA coords with 1e6 + |i-j|*100 fallback for invalid frames. */
export function buildKnnEdges(coords, frames, L, K = Math.min(16, L)) {
  const edges = new Int32Array(L * K);
  const bestD = new Float32Array(K);
  const bestJ = new Int32Array(K);
  for (let i = 0; i < L; i++) {
    bestD.fill(Infinity);
    bestJ.fill(0);
    const vi = frames[i * 13 + 12] > 0;
    const ix = coords[i * 9 + 3], iy = coords[i * 9 + 4], iz = coords[i * 9 + 5];
    for (let j = 0; j < L; j++) {
      let d;
      if (vi && frames[j * 13 + 12] > 0) {
        const dx = f32(ix - coords[j * 9 + 3]), dy = f32(iy - coords[j * 9 + 4]), dz = f32(iz - coords[j * 9 + 5]);
        d = f32(Math.sqrt(dot3(dx, dy, dz, dx, dy, dz)));
        if (!(d < 1e6)) throw new RangeError('Coordinate pairwise distances exceed max supported distance (1e6)');
      } else {
        d = f32(1e6 + Math.abs(i - j) * 100);
      }
      if (d < bestD[K - 1]) {
        let p = K - 1;
        while (p > 0 && d < bestD[p - 1]) {
          bestD[p] = bestD[p - 1];
          bestJ[p] = bestJ[p - 1];
          p--;
        }
        bestD[p] = d;
        bestJ[p] = j;
      }
    }
    edges.set(bestJ, i * K);
  }
  return edges;
}

/** Exact ESM-3 RelativePositionEmbedding indices for KNN neighborhoods (bins=32 -> 1..65). */
export function buildRelativePosIds(edges, residueIndex, L, K, bins = 32) {
  const ids = new Int32Array(L * K);
  for (let i = 0; i < L; i++) {
    const q = residueIndex[edges[i * K]];
    for (let k = 0; k < K; k++) {
      const diff = residueIndex[edges[i * K + k]] - q;
      const clamped = diff < -bins ? -bins : diff > bins ? bins : diff;
      ids[i * K + k] = clamped + bins + 1;
    }
  }
  return ids;
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
export async function loadWeights(baseUrl, { read, signal, onProgress, cache = 'none', onTensor, progressPhase = 'weights' } = {}) {
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
    loaded+=shard.bytes;onProgress?.({phase:progressPhase,loaded,total:manifest.total_bytes,file:shard.file});
  }
  if(pending.size) throw new Error('Incomplete bundle');
  return {manifest,config:manifest.config,weights};
}
