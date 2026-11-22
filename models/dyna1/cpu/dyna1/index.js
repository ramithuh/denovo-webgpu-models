import {
  decodeTensor,
  loadWeights,
  normalizeBackboneCoords,
  normalizeResidueIndex,
  resolvePredictInputs,
  buildStructureFrames,
  buildKnnEdges,
  buildRelativePosIds,
} from '../../shared/dyna1/index.js';

const f32 = Math.fround;
const INV_SQRT3 = f32(1 / Math.sqrt(3));
const softplus = v => v > 20 ? f32(v) : f32(Math.log1p(Math.exp(v)));

export function linear(x,w,b,rows,dim,outDim) {
  const y = new Float32Array(rows*outDim);
  for(let r=0;r<rows;r++) for(let o=0;o<outDim;o++) {
    let s0=0,s1=0,s2=0,s3=0,k=0;const xi=r*dim,wi=o*dim;
    for(;k+3<dim;k+=4) {s0+=x[xi+k]*w[wi+k];s1+=x[xi+k+1]*w[wi+k+1];s2+=x[xi+k+2]*w[wi+k+2];s3+=x[xi+k+3]*w[wi+k+3];}
    let s=s0+s1+s2+s3;for(;k<dim;k++) s+=x[xi+k]*w[wi+k];y[r*outDim+o]=s+(b?.[o]??0);
  }
  return y;
}
export function layerNorm(x,w,b,rows,dim,eps=1e-5) {
  const y=new Float32Array(x.length);
  for(let r=0;r<rows;r++) {const off=r*dim;let mean=0,v=0;for(let j=0;j<dim;j++) mean+=x[off+j];mean/=dim;for(let j=0;j<dim;j++) v+=(x[off+j]-mean)**2;const inv=1/Math.sqrt(v/dim+eps);for(let j=0;j<dim;j++) y[off+j]=(x[off+j]-mean)*inv*w[j]+(b?.[j]??0);}
  return y;
}
function rope(x,L,D,H) {
  const dh=D/H,half=dh/2,out=new Float32Array(x.length);
  for(let p=0;p<L;p++) for(let h=0;h<H;h++) for(let j=0;j<half;j++) {
    const a=f32(p*f32(1/10000**(2*j/dh))),c=f32(Math.cos(a)),s=f32(Math.sin(a)),i=p*D+h*dh+j;
    out[i]=f32(x[i]*c)-f32(x[i+half]*s);out[i+half]=f32(x[i+half]*c)+f32(x[i]*s);
  }
  return out;
}
function attention(q,k,v,L,D,H) {
  const dh=D/H,y=new Float32Array(L*D),p=new Float32Array(L);
  for(let h=0;h<H;h++) for(let i=0;i<L;i++) {
    let max=-Infinity;
    for(let j=0;j<L;j++) {let s=0;for(let d=0;d<dh;d++) s+=q[i*D+h*dh+d]*k[j*D+h*dh+d];p[j]=s/Math.sqrt(dh);max=Math.max(max,p[j]);}
    let sum=0;for(let j=0;j<L;j++) {p[j]=Math.exp(p[j]-max);sum+=p[j];}for(let j=0;j<L;j++) p[j]/=sum;
    for(let d=0;d<dh;d++) {let s=0;for(let j=0;j<L;j++) s+=p[j]*v[j*D+h*dh+d];y[i*D+h*dh+d]=s;}
  }
  return y;
}
function residual(x,y,scale=1) {const out=new Float32Array(x.length);for(let i=0;i<x.length;i++) out[i]=x[i]+f32(y[i]/scale);return out;}

function geometricAttention(z, frames, K, H, B, dw, rw) {
  const R = B * K, vec = new Float32Array(z.length), out = new Float32Array(R * H * 3);
  for (let r = 0; r < R; r++) {
    const fb = r * 13;
    const m00 = frames[fb], m01 = frames[fb+1], m02 = frames[fb+2];
    const m10 = frames[fb+3], m11 = frames[fb+4], m12 = frames[fb+5];
    const m20 = frames[fb+6], m21 = frames[fb+7], m22 = frames[fb+8];
    const tx = frames[fb+9], ty = frames[fb+10], tz = frames[fb+11];
    for (let v = 0; v < H * 5; v++) {
      const o = r * H * 15 + v * 3;
      const vx = z[o], vy = z[o+1], vz = z[o+2];
      let sx = f32(f32(f32(vx * m00) + f32(vy * m01)) + f32(vz * m02));
      let sy = f32(f32(f32(vx * m10) + f32(vy * m11)) + f32(vz * m12));
      let sz = f32(f32(f32(vx * m20) + f32(vy * m21)) + f32(vz * m22));
      if (v >= 3 * H) { sx = f32(sx + tx); sy = f32(sy + ty); sz = f32(sz + tz); }
      vec[o] = sx; vec[o+1] = sy; vec[o+2] = sz;
    }
  }
  const rScale = Float32Array.from(rw, softplus);
  const dScale = Float32Array.from(dw, softplus);
  const prob = new Float32Array(K);
  for (let b = 0; b < B; b++) {
    for (let h = 0; h < H; h++) {
      const rs = rScale[h], ds = dScale[h];
      for (let r = 0; r < K; r++) {
        const ri = b * K + r, qi = ri * H * 15 + h * 3;
        const qrx = vec[qi], qry = vec[qi+1], qrz = vec[qi+2];
        const qdx = vec[qi+H*9], qdy = vec[qi+H*9+1], qdz = vec[qi+H*9+2];
        let max = -Infinity;
        for (let c = 0; c < K; c++) {
          const ci = b * K + c, ki = ci * H * 15 + H * 3 + h * 3;
          const rot = f32(f32(f32(qrx * vec[ki]) + f32(qry * vec[ki+1])) + f32(qrz * vec[ki+2]));
          const ex = f32(qdx - vec[ki+H*9]), ey = f32(qdy - vec[ki+H*9+1]), ez = f32(qdz - vec[ki+H*9+2]);
          const dist2 = f32(f32(f32(ex * ex) + f32(ey * ey)) + f32(ez * ez));
          const bias = frames[ci * 13 + 12] > 0 ? 1.0 : -3.4028234663852886e38;
          const score = f32(f32(f32(rot * INV_SQRT3) * rs) - f32(f32(Math.sqrt(dist2) * INV_SQRT3) * ds)) + bias;
          prob[c] = score;
          if (score > max) max = score;
        }
        let den = 0;
        for (let c = 0; c < K; c++) { prob[c] = Math.exp(prob[c] - max); den += prob[c]; }
        let vx = 0, vy = 0, vz = 0;
        for (let c = 0; c < K; c++) {
          const p = prob[c] / den;
          const vi = (b * K + c) * H * 15 + H * 6 + h * 3;
          vx += p * vec[vi]; vy += p * vec[vi+1]; vz += p * vec[vi+2];
        }
        const fb = ri * 13, ob = (ri * H + h) * 3;
        out[ob]   = f32(f32(f32(vx * frames[fb])   + f32(vy * frames[fb+3])) + f32(vz * frames[fb+6]));
        out[ob+1] = f32(f32(f32(vx * frames[fb+1]) + f32(vy * frames[fb+4])) + f32(vz * frames[fb+7]));
        out[ob+2] = f32(f32(f32(vx * frames[fb+2]) + f32(vy * frames[fb+5])) + f32(vz * frames[fb+8]));
      }
    }
  }
  return out;
}

export function encodeStructureCPU(w, config, coords, { residueIndex, signal, codebookNorms } = {}) {
  const L = coords.length / 9;
  const K = Math.min(config.knn ?? 16, L), D = config.d_model ?? 1024, H = config.geom_heads ?? 128, F = config.ffn_size ?? 4096;
  const layers = config.layers ?? 2, codeDim = config.code_dim ?? 128, codes = config.codebook_size ?? 4096;
  const resIdx = normalizeResidueIndex(residueIndex, L);
  const frames = buildStructureFrames(coords, L);
  const edges = buildKnnEdges(coords, frames, L, K);
  const relIds = buildRelativePosIds(edges, resIdx, L, K, config.relative_position_bins ?? 32);
  const R = L * K;
  const knnFrames = new Float32Array(R * 13);
  for (let i = 0; i < R; i++) knnFrames.set(frames.subarray(edges[i] * 13, edges[i] * 13 + 13), i * 13);
  const relEmbW = w['encoder.relative_positional_embedding.embedding.weight'];
  let x = new Float32Array(R * D);
  for (let i = 0; i < R; i++) x.set(relEmbW.subarray(relIds[i] * D, (relIds[i] + 1) * D), i * D);
  for (let layer = 0; layer < layers; layer++) {
    signal?.throwIfAborted();
    const p = `encoder.transformer.blocks.${layer}`, g = `${p}.geom_attn`;
    const ns = layerNorm(x, w[g + '.s_norm.weight'], w[g + '.s_norm.bias'], R, D);
    const proj = linear(ns, w[g + '.proj.weight'], w[g + '.proj.bias'], R, D, H * 15);
    const ctx = geometricAttention(proj, knnFrames, K, H, L, w[g + '.distance_scale_per_head'], w[g + '.rotation_scale_per_head']);
    x = residual(x, linear(ctx, w[g + '.out_proj.weight'], w[g + '.out_proj.bias'], R, H * 3, D));
    const ffnIn = layerNorm(x, w[p + '.ffn.0.weight'], w[p + '.ffn.0.bias'], R, D);
    const ff = linear(ffnIn, w[p + '.ffn.1.weight'], w[p + '.ffn.1.bias'], R, D, 2 * F);
    const act = new Float32Array(R * F);
    for (let r = 0; r < R; r++) for (let j = 0; j < F; j++) {
      const z = ff[r * 2 * F + j];
      act[r * F + j] = f32(z / (1 + Math.exp(-z))) * ff[r * 2 * F + F + j];
    }
    x = residual(x, linear(act, w[p + '.ffn.3.weight'], w[p + '.ffn.3.bias'], R, F, D));
  }
  const selected = new Float32Array(L * D);
  for (let i = 0; i < L; i++) if (frames[i * 13 + 12] > 0) selected.set(x.subarray(i * K * D, i * K * D + D), i * D);
  const preVq = linear(selected, w['encoder.pre_vq_proj.weight'], w['encoder.pre_vq_proj.bias'], L, D, codeDim);
  const cb = w['encoder.codebook.embeddings'];
  let norms = codebookNorms;
  if (!norms) {
    norms = new Float32Array(codes);
    for (let k = 0; k < codes; k++) { let s = 0; for (let d = 0; d < codeDim; d++) s += cb[k * codeDim + d] ** 2; norms[k] = s; }
  }
  const tokens = new Int32Array(L);
  for (let i = 0; i < L; i++) {
    let zn = 0;
    for (let d = 0; d < codeDim; d++) zn += preVq[i * codeDim + d] ** 2;
    let best = Infinity, idx = 0;
    for (let k = 0; k < codes; k++) {
      let s = 0;
      for (let d = 0; d < codeDim; d++) s += preVq[i * codeDim + d] * cb[k * codeDim + d];
      const dist = f32(f32(f32(zn) - f32(2 * f32(s))) + norms[k]);
      if (dist < best) { best = dist; idx = k; }
    }
    tokens[i] = idx;
  }
  return { structureTokens: tokens, preVq, frames, edges };
}

function decodeWeights(weights) {
  const out = {};
  for (const [k, v] of Object.entries(weights)) out[k] = v instanceof Float32Array ? v : decodeTensor(v);
  return out;
}

function computeCodebookNorms(cb, codes, dim) {
  const norms = new Float32Array(codes);
  for (let k = 0; k < codes; k++) { let s = 0; for (let d = 0; d < dim; d++) s += cb[k * dim + d] ** 2; norms[k] = s; }
  return norms;
}

export class Dyna1CPU {
  constructor(bundle, { structureBundle = null, structureEncoderUrl = null, loadOptions = {} } = {}) {
    this.config = bundle.config;
    this.w = decodeWeights(bundle.weights);
    this.structureEncoderUrl = structureEncoderUrl;
    this.loadOptions = loadOptions;
    this.structW = null;
    this.structConfig = null;
    this.codebookNorms = null;
    this._structPromise = null;
    if (structureBundle) this._setStructureBundle(structureBundle);
  }
  _setStructureBundle(bundle) {
    this.structConfig = bundle.config;
    this.structW = decodeWeights(bundle.weights);
    this.codebookNorms = computeCodebookNorms(
      this.structW['encoder.codebook.embeddings'],
      this.structConfig.codebook_size ?? 4096,
      this.structConfig.code_dim ?? 128
    );
  }
  async preloadStructureEncoder({ onProgress, signal } = {}) {
    if (this.structW) return;
    if (!this.structureEncoderUrl) throw new Error('structureEncoderUrl is not configured');
    if (!this._structPromise) {
      this._structPromise = loadWeights(this.structureEncoderUrl, {
        ...this.loadOptions,
        onProgress: onProgress ?? this.loadOptions.onProgress,
        signal: signal ?? this.loadOptions.signal,
        progressPhase: 'structure-encoder',
      }).then(bundle => {
        this._setStructureBundle(bundle);
      }).catch(err => {
        this._structPromise = null;
        throw err;
      });
    }
    await this._structPromise;
  }
  encodeStructure(structureCoords, { residueIndex, signal } = {}) {
    if (!this.structW) throw new Error('Structure encoder not loaded; call await engine.preloadStructureEncoder() first');
    const start = performance.now();
    const { coords } = normalizeBackboneCoords(structureCoords, this.config.max_sequence_length);
    const { structureTokens, preVq } = encodeStructureCPU(this.structW, this.structConfig, coords, {
      residueIndex,
      signal,
      codebookNorms: this.codebookNorms,
    });
    return { structureTokens, preVq, timingMs: performance.now() - start };
  }
  predict(sequence, opts = {}) {
    const { returnIntermediates = false, signal } = opts;
    const start = performance.now();
    const resolved = resolvePredictInputs(sequence, opts, this.config.max_sequence_length);
    const L = resolved.length, D = this.config.d_model, w = this.w;
    const tokens = resolved.seqTokens;
    const traces = { tokens };
    const save = (n, v) => { if (returnIntermediates) traces[n] = v; };
    const ln = (x, p, eps = 1e-5) => layerNorm(x, w[p + '.weight'], w[p + '.bias'], L, D, eps);
    const lin = (x, p, di = D, do_ = D) => linear(x, w[p + '.weight'], w[p + '.bias'], L, di, do_);
    const split = qkv => {
      const q = new Float32Array(L * D), k = new Float32Array(L * D), v = new Float32Array(L * D);
      for (let i = 0; i < L; i++) {
        q.set(qkv.subarray(i * 3 * D, i * 3 * D + D), i * D);
        k.set(qkv.subarray(i * 3 * D + D, i * 3 * D + 2 * D), i * D);
        v.set(qkv.subarray(i * 3 * D + 2 * D, (i + 1) * 3 * D), i * D);
      }
      return [q, k, v];
    };
    let structureTokens = resolved.structureTokens;
    if (resolved.mode !== 'seq-only') {
      if (!this.structW) throw new Error('Structure encoder not loaded; call await engine.preloadStructureEncoder() first');
      if (!structureTokens) {
        const enc = encodeStructureCPU(this.structW, this.structConfig, resolved.coords, {
          residueIndex: resolved.residueIndex,
          signal,
          codebookNorms: this.codebookNorms,
        });
        structureTokens = enc.structureTokens;
        save('pre_vq', enc.preVq);
      }
      save('structure_tokens', structureTokens);
    }
    let x = new Float32Array(L * D);
    const embW = w['embedding.weight'];
    if (resolved.mode === 'seq-only') {
      for (let i = 0; i < L; i++) x.set(embW.subarray(tokens[i] * D, (tokens[i] + 1) * D), i * D);
    } else {
      const deltaW = this.structW['structure_tokens_embed_delta'];
      for (let i = 0; i < L; i++) {
        const sb = tokens[i] * D, tb = structureTokens[i] * D, ob = i * D;
        for (let d = 0; d < D; d++) x[ob + d] = embW[sb + d] + deltaW[tb + d];
      }
    }
    save('embedding', x);
    for (let i = 0; i < this.config.backbone_layers; i++) {
      signal?.throwIfAborted();
      const p = `backbone.${i}`, F = this.config.ffn_size, H = this.config.backbone_heads;
      let [q, k, v] = split(lin(ln(x, p + '.attn.layernorm_qkv.0'), p + '.attn.layernorm_qkv.1', D, 3 * D));
      q = rope(ln(q, p + '.attn.q_ln'), L, D, H); k = rope(ln(k, p + '.attn.k_ln'), L, D, H);
      x = residual(x, lin(attention(q, k, v, L, D, H), p + '.attn.out_proj'), this.config.residual_scale);
      const a = lin(ln(x, p + '.ffn.0'), p + '.ffn.1', D, 2 * F), act = new Float32Array(L * F);
      for (let r = 0; r < L; r++) for (let j = 0; j < F; j++) {
        const z = a[r * 2 * F + j];
        act[r * F + j] = f32(z / (1 + Math.exp(-z))) * a[r * 2 * F + F + j];
      }
      x = residual(x, lin(act, p + '.ffn.3', F, D), this.config.residual_scale); save(`backbone.${i}`, x);
    }
    save('final_embeddings', x); x = ln(x, 'head.norm1', 1e-12);
    for (let i = 0; i < this.config.head_layers; i++) {
      signal?.throwIfAborted();
      const p = `head.encoder.layers.${i}`, F = this.config.head_ffn_size;
      const [q, k, v] = split(linear(ln(x, p + '.norm1'), w[p + '.self_attn.in_proj_weight'], w[p + '.self_attn.in_proj_bias'], L, D, 3 * D));
      x = residual(x, lin(attention(q, k, v, L, D, this.config.head_heads), p + '.self_attn.out_proj'));
      const a = lin(ln(x, p + '.norm2'), p + '.linear1', D, F); for (let j = 0; j < a.length; j++) a[j] = Math.max(0, a[j]);
      x = residual(x, lin(a, p + '.linear2', F, D)); save(`head.${i}`, x);
    }
    x = ln(x, 'head.norm2', 1e-12);
    const logits = lin(x, 'head.output_layer', D, 1), probs = Float32Array.from(logits, z => 1 / (1 + Math.exp(-(f32(z - f32(Math.log(.05 / .95)))))));
    save('logits', logits); save('probs', probs);
    return {
      logits,
      probs,
      residues: resolved.residues,
      ...(structureTokens ? { structureTokens } : {}),
      mode: resolved.mode,
      timingMs: performance.now() - start,
      ...(returnIntermediates ? { intermediates: traces } : {}),
    };
  }
  destroy() { this.w = {}; this.structW = null; this.codebookNorms = null; this._structPromise = null; }
}

export async function createDyna1CPU(options) {
  const bundle = await loadWeights(options.baseUrl, options);
  const inferredStructUrl = options.structureEncoderUrl ??
    `${options.baseUrl.replace(/\/$/, '').replace(/\/(fp32|fp16|int8)$/, '')}/structure_encoder`;
  const cpu = new Dyna1CPU(bundle, {
    structureEncoderUrl: inferredStructUrl,
    loadOptions: { read: options.read, cache: options.cache, onProgress: options.onProgress },
  });
  if (options.preloadStructureEncoder) {
    await cpu.preloadStructureEncoder({ onProgress: options.onProgress, signal: options.signal });
  }
  return cpu;
}
