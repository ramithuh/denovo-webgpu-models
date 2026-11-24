// Handwritten module hierarchy, transcribed from pinned LD1, LD4-LD7, AE1, and AE3 (encoder + decoder).
// Shapes come from runtime tensors; no recorded operation graphs or step traces.
import { abort } from './loader.js';
import constants from './constants.js';
import {
  structSeqFeaturesCPU,
  ae3PairFeaturesCPU,
  ld4MotifPairFeaturesCPU,
  padPairCPU,
  expandSplitRowsCPU,
} from './featurizer.js';

export class LaProteinaNetwork {
  constructor(ops, weights, config, model) {
    this.o = ops;
    this.w = weights;
    this.model = model;
    this.isLD = model.startsWith('LD');
    this.isEncoder = model === 'AE3_encoder';
    this.c = this.isLD ? config : this.isEncoder ? config.encoder : config.decoder;
  }
  linear(x, p) {
    if (!this.w[p + '.weight']) throw Error(`Missing parameter ${p}.weight`);
    return this.o.linear(x, this.w[p + '.weight'], this.w[p + '.bias']);
  }
  norm(x, p) {
    return this.o.norm(x, this.w[p + '.weight'], this.w[p + '.bias']);
  }
  transition(x, p) {
    const o = this.o, h = this.linear(x, p + '.swish_linear.0');
    if (o.fastSwiglu) return this.linear(o.fastSwiglu(h), p + '.linear_out');
    const n = h.shape.at(-1) / 2;
    const a = o.slice(h, h.shape.length - 1, 0, n), gate = o.slice(h, h.shape.length - 1, n, n * 2);
    return this.linear(o.mul(a, o.silu(gate)), p + '.linear_out');
  }
  adaptive(x, c, p, splitRow = 0) {
    const o = this.o, cond = this.norm(c, p + '.norm_cond');
    let gammaRaw = this.linear(cond, p + '.to_gamma.0');
    let beta = this.linear(cond, p + '.to_beta');
    if (o.fastAdaptive) return o.fastAdaptive(x, gammaRaw, beta, 1e-5, splitRow);
    if (splitRow > 0) {
      gammaRaw = expandSplitRowsCPU(o, gammaRaw, splitRow, x.shape[0]);
      beta = expandSplitRowsCPU(o, beta, splitRow, x.shape[0]);
    }
    const normed = o.norm(x);
    return o.add(o.mul(normed, o.sigmoid(gammaRaw)), beta);
  }
  scale(x, c, p, splitRow = 0) {
    let gammaRaw = this.linear(c, p + '.to_adaln_zero_gamma.0');
    if (this.o.fastScale) return this.o.fastScale(x, gammaRaw, splitRow);
    if (splitRow > 0) gammaRaw = expandSplitRowsCPU(this.o, gammaRaw, splitRow, x.shape[0]);
    return this.o.mul(x, this.o.sigmoid(gammaRaw));
  }
  residualScale(seq, update, c, p, splitRow = 0) {
    let gammaRaw = this.linear(c, p + '.to_adaln_zero_gamma.0');
    if (this.o.fastResidualScaleAdd) return this.o.fastResidualScaleAdd(seq, update, gammaRaw, splitRow);
    if (splitRow > 0) gammaRaw = expandSplitRowsCPU(this.o, gammaRaw, splitRow, seq.shape[0]);
    return this.o.add(seq, this.o.mul(update, this.o.sigmoid(gammaRaw)));
  }
  time(tCa, tZ) {
    const o = this.o, D = this.c.t_emb_dim;
    const times = o.tensor([2], [tCa, tZ]);
    return o.kernel(
      [1, 2 * D],
      [times],
      `const d=i%${D},k=d%${D / 2},f=Math.fround,e=rfExp(f(k*f(-Math.log(2000)/${D / 2 - 1}))),v=f(f(a0[Math.floor(i/${D})]*2000)*e);o[i]=d<${D / 2}?Math.sin(v):Math.cos(v);`,
      `let d=i%${D}u;let k=d%${D / 2}u;let e=rfExp(f32(k)*${Math.fround(-Math.log(2000) / (D / 2 - 1))});let v=(a0[i/${D}u]*2000.0)*e;let first=(a0[i/${D}u]*2000.0)*rfExp(f32(k-k%4u)*${Math.fround(-Math.log(2000) / (D / 2 - 1))});let sc=rfSinCos(v,first>=125.0);o[i]=select(sc.y,sc.x,d<${D / 2}u);`,
    );
  }
  distances(ca) {
    const o = this.o, L = ca.shape[0];
    return o.kernel(
      [L, L],
      [ca, o.tensor([29], constants.distance_bins)],
      `const r=Math.floor(i/${L})*3,c=i%${L}*3,f=Math.fround;let s=0;for(let k=0;k<3;k++){const d=f(a0[r+k]-a0[c+k]);s=f(s+f(d*d));}const d=f(Math.sqrt(s));let bin=0;for(let k=0;k<29;k++)if(d>a1[k])bin++;o[i]=bin;`,
      `let r=(i/${L}u)*3u;let c=(i%${L}u)*3u;var s=0.0;for(var k=0u;k<3u;k++){let d=a0[r+k]-a0[c+k];s=s+d*d;}let d=sqrt(s);var bin=0u;for(var k=0u;k<29u;k++){if(d>a1[k]){bin++;}}o[i]=f32(bin);`,
    );
  }
  pairFeatures(ca, scCa) {
    const o = this.o, L = ca.shape[0], W = this.isLD ? 217 : 157;
    const dist = this.distances(ca), inputs = [dist];
    if (scCa) inputs.push(this.distances(scCa));
    return o.kernel(
      [L, L, W],
      inputs,
      `const pair=Math.floor(i/${W}),k=i%${W},r=Math.floor(pair/${L}),c=pair%${L},sep=Math.max(0,Math.min(126,r-c+63));o[i]=k<127?Number(k===sep):k<157?Number(k-127===a0[pair]):${scCa ? 'k<187?Number(k-157===a1[pair]):0' : '0'};`,
      `let pair=i/${W}u;let k=i%${W}u;let r=pair/${L}u;let c=pair%${L}u;let sep=u32(clamp(i32(r)-i32(c)+63,0,126));var v=0.0;if(k<127u){v=select(0.0,1.0,k==sep);}else if(k<157u){v=select(0.0,1.0,k-127u==u32(a0[pair]));}${scCa ? 'else if(k<187u){v=select(0.0,1.0,k-157u==u32(a1[pair]));}' : ''}o[i]=v;`,
    );
  }
  prepareMotif(cond) {
    const o = this.o, L = cond.length;
    const xMotifT = o.upload([L, 37, 3], cond.xMotif);
    const motifMaskT = o.upload([L, 37], cond.motifMask);
    const seqMotifT = o.upload([L], cond.seqMotif);
    if (this.model === 'LD4') {
      const seqFeat = o.structSeqFeatures
        ? o.structSeqFeatures(xMotifT, motifMaskT, seqMotifT, null, 1, 1.0)
        : structSeqFeaturesCPU(o, xMotifT, motifMaskT, seqMotifT, null, 1, 1.0);
      const pairBins = o.ld4MotifBins ? o.ld4MotifBins(xMotifT, motifMaskT) : null;
      const pairFeatCPU = !o.ld4MotifBins ? ld4MotifPairFeaturesCPU(o, xMotifT, motifMaskT) : null;
      return { length: L, seqFeat, pairBins, pairFeatCPU, keep: [seqFeat, pairBins, pairFeatCPU].filter(Boolean) };
    }
    if (this.model === 'LD5') {
      const seqFeat = o.structSeqFeatures
        ? o.structSeqFeatures(xMotifT, motifMaskT, seqMotifT, null, 2, 1.0)
        : structSeqFeaturesCPU(o, xMotifT, motifMaskT, seqMotifT, null, 2, 1.0);
      return { length: L, seqFeat, keep: [seqFeat] };
    }
    if (this.model === 'LD6' || this.model === 'LD7') {
      const M = cond.activeResidueIndices.length;
      if (M === 0) return { length: L, M: 0, uidxTokens: null, keep: [] };
      const activeIdxT = o.upload([M], Float32Array.from(cond.activeResidueIndices));
      const uidxFeats = o.structSeqFeatures
        ? o.structSeqFeatures(xMotifT, motifMaskT, seqMotifT, activeIdxT, 3, 1.0)
        : structSeqFeaturesCPU(o, xMotifT, motifMaskT, seqMotifT, activeIdxT, 3, 1.0);
      const uidxTokens = this.linear(uidxFeats, 'motif_uidx_factory.linear_out');
      return { length: L, M, uidxTokens, keep: [uidxTokens] };
    }
    throw Error(`Model ${this.model} does not accept motif conditioning`);
  }
  attention(x, pair, c, p, pairStats = null, splitRow = 0) {
    const o = this.o, N = x.shape[0], H = this.c.nheads, D = this.c.token_dim / H;
    x = this.adaptive(x, c, p + '.adaln', splitRow);
    const prefix = p + '.mha', node = this.norm(x, prefix + '.node_norm');
    const qkv = this.linear(node, prefix + '.to_qkv'), C = this.c.token_dim;
    if (o.attnLogits && o.attnContextGated && pairStats) {
      const q = o.normStrided(qkv, N, C, 3 * C, 0, this.w[prefix + '.q_layer_norm.weight'], this.w[prefix + '.q_layer_norm.bias']);
      const k = o.normStrided(qkv, N, C, 3 * C, C, this.w[prefix + '.k_layer_norm.weight'], this.w[prefix + '.k_layer_norm.bias']);
      const biasPerm = o.pairBiasPermute(pair, pairStats, this.w[prefix + '.pair_norm.weight'], this.w[prefix + '.pair_norm.bias'], this.w[prefix + '.to_bias.weight'], this.w[prefix + '.to_bias.bias'], N, H);
      const sim = o.attnLogits(q, k, biasPerm, N, H, D);
      const probs = o.softmax(sim);
      const gate = this.linear(node, prefix + '.to_g');
      const update = o.attnContextGated(probs, qkv, gate, N, H, D);
      return this.linear(update, prefix + '.to_out_node');
    }
    const q = o.reshape(this.norm(o.slice(qkv, 1, 0, C), prefix + '.q_layer_norm'), [N, H, D]);
    const k = o.reshape(this.norm(o.slice(qkv, 1, C, 2 * C), prefix + '.k_layer_norm'), [N, H, D]);
    const v = o.reshape(o.slice(qkv, 1, 2 * C, 3 * C), [N, H, D]);
    const bias = this.linear(this.norm(pair, prefix + '.pair_norm'), prefix + '.to_bias');
    const sim = o.add(o.mul(o.einsum('ihd,jhd->hij', q, k), 1 / Math.sqrt(D)), o.permute(bias, [2, 0, 1]));
    const probs = o.softmax(sim);
    let update = o.reshape(o.einsum('hij,jhd->ihd', probs, v), [N, C]);
    update = o.mul(update, o.sigmoid(this.linear(node, prefix + '.to_g')));
    return this.scale(this.linear(update, prefix + '.to_out_node'), c, p + '.scale_output', splitRow);
  }
  async forward(input, { signal, onProgress, trace } = {}) {
    const o = this.o;
    const { ca, z, scCa, scZ, motifState, eps, keep: extraKeep } = input;
    const L = this.isEncoder ? input.coordsNm.shape[0] : ca.shape[0];
    const keep = [ca, z, scCa, scZ, eps, ...(motifState?.keep ?? []), ...(extraKeep ?? [])].filter(Boolean);
    const take = async (name, t) => {
      if (trace) trace[name] = Array.from(await o.read(t));
    };
    abort(signal);
    let c, seq, pair, splitRow = 0;

    if (this.isLD) {
      const t = this.time(input.tCa, input.tZ);
      c = this.linear(t, 'cond_factory.linear_out');
      c = this.transition(this.transition(c, 'transition_c_1'), 'transition_c_2');

      if (this.model === 'LD1') {
        const features = o.cat([ca, z, scCa ?? o.constant([L, 3]), scZ ?? o.constant([L, 8]), o.constant([L, 24])]);
        seq = this.linear(features, 'init_repr_factory.linear_out');
      } else {
        if (!motifState || motifState.length !== L) throw Error(`${this.model} requires prepared motif conditioning of length ${L}`);
        const baseParts = [ca, z, scCa ?? o.constant([L, 3]), scZ ?? o.constant([L, 8]), o.constant([L, 23])];
        if (this.model === 'LD4' || this.model === 'LD5') baseParts.push(motifState.seqFeat);
        seq = this.linear(o.cat(baseParts), 'init_repr_factory.linear_out');
      }

      let pLin;
      if (this.model === 'LD4') {
        pLin = o.ld4PairLinear
          ? o.ld4PairLinear(ca, scCa, motifState.pairBins, this.w['pair_repr_builder.init_repr_factory.linear_out.weight'], this.w['pair_repr_builder.init_repr_factory.linear_out.bias'])
          : this.linear(o.cat([this.pairFeatures(ca, scCa), motifState.pairFeatCPU], 2), 'pair_repr_builder.init_repr_factory.linear_out');
      } else {
        pLin = o.pairLinearInit
          ? o.pairLinearInit(ca, scCa, this.w['pair_repr_builder.init_repr_factory.linear_out.weight'], this.w['pair_repr_builder.init_repr_factory.linear_out.bias'])
          : this.linear(this.pairFeatures(ca, scCa), 'pair_repr_builder.init_repr_factory.linear_out');
      }
      pair = this.norm(pLin, 'pair_repr_builder.init_repr_factory.ln_out');
      const pc = this.norm(this.linear(t, 'pair_repr_builder.cond_factory.linear_out'), 'pair_repr_builder.cond_factory.ln_out');
      pair = this.adaptive(pair, pc, 'pair_repr_builder.adaln');

      if ((this.model === 'LD6' || this.model === 'LD7') && motifState.M > 0) {
        const N = L + motifState.M;
        seq = o.cat([seq, motifState.uidxTokens], 0);
        c = o.cat([c, o.constant([1, this.c.dim_cond], 0)], 0);
        pair = o.padPair ? o.padPair(pair, L, N) : padPairCPU(o, pair, L, N);
        splitRow = L;
      }
    } else if (this.isEncoder) {
      c = o.constant([1, this.c.dim_cond]);
      c = this.transition(this.transition(c, 'transition_c_1'), 'transition_c_2');
      const seqFeats = o.structSeqFeatures
        ? o.structSeqFeatures(input.coordsNm, input.coordMask, input.residueTypes, input.chainBreaks, 0, 10.0)
        : structSeqFeaturesCPU(o, input.coordsNm, input.coordMask, input.residueTypes, input.chainBreaks, 0, 10.0);
      seq = this.linear(seqFeats, 'init_repr_factory.linear_out');
      pair = o.ae3PairLinear
        ? o.ae3PairLinear(input.coordsNm, input.coordMask, this.w['pair_rep_factory.linear_out.weight'], this.w['pair_rep_factory.linear_out.bias'])
        : this.linear(ae3PairFeaturesCPU(o, input.coordsNm, input.coordMask), 'pair_rep_factory.linear_out');
    } else {
      c = o.constant([1, this.c.dim_cond]);
      c = this.transition(this.transition(c, 'transition_c_1'), 'transition_c_2');
      seq = this.linear(o.cat([ca, z]), 'init_repr_factory.linear_out');
      pair = o.pairLinearInit
        ? o.pairLinearInit(ca, null, this.w['pair_rep_factory.linear_out.weight'], this.w['pair_rep_factory.linear_out.bias'])
        : this.linear(this.pairFeatures(ca), 'pair_rep_factory.linear_out');
    }

    await take('init', seq);
    await take('cond', c);
    await take('pair', pair);
    const pairStats = o.rowStats ? o.rowStats(pair) : null;
    const fastAttn = Boolean(o.attnLogits && o.attnContextGated && pairStats);
    const phaseName = this.model.toLowerCase();
    for (let i = 0; i < this.c.nlayers; i++) {
      abort(signal);
      const p = `transformer_layers.${i}`;
      if (fastAttn) {
        const attnOut = this.attention(seq, pair, c, p + '.mhba', pairStats, splitRow);
        seq = this.residualScale(seq, attnOut, c, p + '.mhba.scale_output', splitRow);
      } else {
        seq = o.add(seq, this.attention(seq, pair, c, p + '.mhba', null, splitRow));
      }
      const tr = this.transition(this.adaptive(seq, c, p + '.transition.adaln', splitRow), p + '.transition.transition');
      seq = this.residualScale(seq, tr, c, p + '.transition.scale_output', splitRow);
      await take(`layer.${i}`, seq);
      onProgress?.({ phase: phaseName, layer: i + 1, total: this.c.nlayers });
      await o.checkpoint([...keep, c, pair, pairStats, seq]);
    }

    if (this.isLD) {
      if (splitRow > 0) seq = o.slice(seq, 0, 0, L);
      return {
        ca: this.linear(this.norm(seq, 'ca_linear.0'), 'ca_linear.1'),
        z: this.linear(this.norm(seq, 'local_latents_linear.0'), 'local_latents_linear.1'),
      };
    }
    if (this.isEncoder) {
      const flat = this.linear(this.norm(seq, 'latent_decoder_mean_n_log_scale.0'), 'latent_decoder_mean_n_log_scale.1');
      const Z = this.c.latent_z_dim ?? 8;
      const mean = o.slice(flat, 1, 0, Z);
      const logScale = o.slice(flat, 1, Z, 2 * Z);
      const zOut = eps ? o.add(mean, o.mul(eps, o.element('exp($0)', logScale))) : mean;
      return { mean, logScale, z: zOut };
    }
    const logits = this.linear(this.norm(seq, 'logit_linear.0'), 'logit_linear.1');
    let coords = o.reshape(this.linear(this.norm(seq, 'struct_linear.0'), 'struct_linear.1'), [L, 37, 3]);
    coords = o.kernel(
      [L, 37, 3],
      [coords, ca],
      `const a=Math.floor(i/3)%37,base=Math.floor(i/111)*3+i%3;o[i]=a===1?a1[base]:${this.c.abs_coors ? 'a0[i]' : 'Math.fround(a0[i]+a1[base])'};`,
      `let a=(i/3u)%37u;let base=(i/111u)*3u+i%3u;o[i]=select(${this.c.abs_coors ? 'a0[i]' : 'a0[i]+a1[base]'},a1[base],a==1u);`,
    );
    return { coords, logits };
  }
}
