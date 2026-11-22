import {
  cbeta,
  cdist,
  rbf,
  normLast,
  quaternionRotation,
  sphericalHarmonics,
} from "./geometry.js";
import { abort } from "./loader.js";
import { backbone } from "./backbone.js";
export class RoseTTAFold {
  constructor(ops, weights, config, constants) {
    this.o = ops;
    this.w = weights;
    this.c = config;
    this.constants = constants;
  }
  linear(x, p) {
    return this.o.linear(x, this.w[p + ".weight"], this.w[p + ".bias"]);
  }
  norm(x, p) {
    return this.o.norm(x, this.w[p + ".weight"], this.w[p + ".bias"]);
  }
  ff(x, p) {
    const o = this.o;
    return this.linear(
      o.relu(this.linear(this.norm(x, p + ".norm"), p + ".linear1")),
      p + ".linear2",
    );
  }
  axial(pair, bias, p, row) {
    const o = this.o,
      L = pair.shape[0],
      H = this.w[p + ".to_b.weight"].shape[0],
      D = this.w[p + ".to_q.weight"].shape[0] / H;
    if (row) {
      pair = o.permute(pair, [1, 0, 2]);
      bias = o.permute(bias, [1, 0, 2]);
    }
    pair = this.norm(pair, p + ".norm_pair");
    bias = this.norm(bias, p + ".norm_bias");
    const q = o.mul(
        o.reshape(this.linear(pair, p + ".to_q"), [L, L, H, D]),
        1 / Math.sqrt(D),
      ),
      k = o.div(
        o.reshape(this.linear(pair, p + ".to_k"), [L, L, H, D]),
        Math.sqrt(L),
      ),
      v = o.reshape(this.linear(pair, p + ".to_v"), [L, L, H, D]);
    let att = o.softmax(
      o.add(o.einsum("nihk,njhk->ijh", q, k), this.linear(bias, p + ".to_b")),
      1,
    );
    let out = o.reshape(o.einsum("ijh,kjhd->ikhd", att, v), [L, L, H * D]);
    out = this.linear(
      o.mul(out, o.sigmoid(this.linear(pair, p + ".to_g"))),
      p + ".to_out",
    );
    return row ? o.permute(out, [1, 0, 2]) : out;
  }
  pairBlock(pair, dist, p) {
    const o = this.o,
      bias = this.linear(
        o.relu(this.linear(dist, p + ".emb_rbf")),
        p + ".proj_rbf",
      );
    pair = o.add(pair, this.axial(pair, bias, p + ".row_attn", true));
    pair = o.add(pair, this.axial(pair, bias, p + ".col_attn", false));
    return o.add(pair, this.ff(pair, p + ".ff"));
  }
  rowMSA(msa, pair, p) {
    const o = this.o,
      L = msa.shape[0],
      H = this.c.n_head_msa,
      D = this.w[p + ".to_q.weight"].shape[0] / H;
    msa = this.norm(msa, p + ".norm_msa");
    pair = this.norm(pair, p + ".norm_pair");
    // In inference N=1; SequenceWeight's softmax over the MSA axis is identically 1.
    const q = o.reshape(this.linear(msa, p + ".to_q"), [L, H, D]),
      k = o.mul(
        o.reshape(this.linear(msa, p + ".to_k"), [L, H, D]),
        1 / Math.sqrt(D),
      ),
      v = o.reshape(this.linear(msa, p + ".to_v"), [L, H, D]);
    const att = o.softmax(
      o.add(o.einsum("qhd,khd->qkh", q, k), this.linear(pair, p + ".to_b")),
      1,
    );
    return this.linear(
      o.mul(
        o.reshape(o.einsum("qkh,khd->qhd", att, v), [L, H * D]),
        o.sigmoid(this.linear(msa, p + ".to_g")),
      ),
      p + ".to_out",
    );
  }
  colMSA(msa, p, global) {
    const o = this.o,
      L = msa.shape[0];
    msa = this.norm(msa, p + ".norm_msa");
    let value = this.linear(msa, p + ".to_v");
    if (global) {
      const H = this.c.n_head_msa,
        D = value.shape[1];
      value = o.reshape(
        o.expr([L, H, D], [o.reshape(value, [L, 1, D])], "$0"),
        [L, H * D],
      );
    }
    return this.linear(
      o.mul(value, o.sigmoid(this.linear(msa, p + ".to_g"))),
      p + ".to_out",
    );
  }
  msaBlock(msa, pair, dist, state, p, global) {
    const o = this.o;
    let bias = this.linear(
      o.cat([this.norm(pair, p + ".norm_pair"), dist]),
      p + ".proj_pair",
    );
    msa = o.add(
      msa,
      this.linear(this.norm(state, p + ".norm_state"), p + ".proj_state"),
    );
    msa = o.add(msa, this.rowMSA(msa, bias, p + ".row_attn"));
    msa = o.add(msa, this.colMSA(msa, p + ".col_attn", global));
    return o.add(msa, this.ff(msa, p + ".ff"));
  }
  msa2pair(msa, pair, p) {
    const o = this.o,
      L = msa.shape[0];
    msa = this.norm(msa, p + ".norm");
    const left = this.linear(msa, p + ".proj_left"),
      right = this.linear(msa, p + ".proj_right");
    return o.add(
      pair,
      this.linear(
        o.reshape(o.einsum("li,mj->lmij", left, right), [L, L, -1]),
        p + ".proj_out",
      ),
    );
  }
  sc(msa, state, p) {
    const o = this.o;
    let x = o.add(
      this.linear(this.norm(msa, p + ".norm_s0"), p + ".linear_s0"),
      this.linear(this.norm(state, p + ".norm_si"), p + ".linear_si"),
    );
    x = o.relu(x);
    x = o.add(
      x,
      this.linear(o.relu(this.linear(x, p + ".linear_1")), p + ".linear_2"),
    );
    x = o.relu(x);
    x = o.add(
      x,
      this.linear(o.relu(this.linear(x, p + ".linear_3")), p + ".linear_4"),
    );
    return o.reshape(this.linear(o.relu(x), p + ".linear_out"), [
      msa.shape[0],
      10,
      2,
    ]);
  }
  radial(edge, p) {
    const o = this.o;
    let x = o.relu(this.norm(this.linear(edge, p + ".net.0"), p + ".net.1"));
    x = o.relu(this.norm(this.linear(x, p + ".net.3"), p + ".net.4"));
    return this.linear(x, p + ".net.6");
  }
  basis(relative) {
    const o = this.o,
      E = relative.shape[0],
      sh = sphericalHarmonics(o, relative),
      result = {};
    for (const [name, entry] of Object.entries(this.constants.basis)) {
      const coeff = o.tensor(entry.shape, entry.data);
      const out = o.einsum("es,slfk->elfk", sh, coeff);
      result[name] = out;
    }
    return result;
  }
  conv(node, edge, basis, src, dst, p, pool, L, mask) {
    const o = this.o,
      E = src.size;
    let out;
    if (this.w[p + ".conv_in.0.radial_func.net.0.weight"]) {
      let acc = null;
      for (let d = 0; d < 2; d++) {
        const f = o.gather(node[d], src),
          b = basis["in" + d],
          freq = b.shape[2],
          C = f.shape[1],
          radial = this.radial(edge, p + `.conv_in.${d}.radial_func`),
          O = radial.shape[1] / (C * freq),
          tmp = o.reshape(o.einsum("ecl,elfk->ecfk", f, b), [E, C * freq, 4]);
        const v = o.einsum(
          "eoc,eck->eok",
          o.reshape(radial, [E, O, C * freq]),
          tmp,
        );
        acc = acc ? o.add(acc, v) : v;
      }
      out = acc;
    } else {
      out = {};
      const f = o.cat([o.gather(node[0], src), o.gather(node[1], src)], 2);
      for (let d = 0; d < 2; d++) {
        const b = basis["out" + d],
          freq = b.shape[2],
          C = f.shape[1],
          radial = this.radial(edge, p + `.conv_out.${d}.radial_func`),
          O = radial.shape[1] / (C * freq),
          tmp = o.reshape(o.einsum("ecl,elfk->ecfk", f, b), [
            E,
            C * freq,
            2 * d + 1,
          ]);
        out[d] = o.einsum(
          "eoc,eck->eok",
          o.reshape(radial, [E, O, C * freq]),
          tmp,
        );
      }
    }
    if (pool)
      for (let d = 0; d < 2; d++) {
        out[d] = o.add(
          out[d],
          o.einsum(
            "oc,eck->eok",
            this.w[p + `.to_kernel_self.${d}`],
            o.gather(node[d], dst),
          ),
        );
        out[d] = o.mul(out[d], o.reshape(mask, [E, 1, 1]));
        out[d] = o.einsum(
          "abck->bck",
          o.reshape(out[d], [L, L, ...out[d].shape.slice(1)]),
        );
      }
    return out;
  }
  se3(node, edge, CA, p, topk, indices) {
    const o = this.o,
      L = CA.shape[0],
      E = L * L,
      src = o.tensor(
        [E],
        Array.from({ length: E }, (_, i) => Math.floor(i / L)),
      ),
      dst = o.tensor(
        [E],
        Array.from({ length: E }, (_, i) => i % L),
      );
    const relative = o.sub(o.gather(CA, dst), o.gather(CA, src)),
      basis = this.basis(relative),
      mask = this.graphMask(CA, indices, topk);
    edge = o.cat([
      o.reshape(edge, [E, edge.shape.at(-1)]),
      normLast(o, relative),
    ]);
    const fused = this.conv(
        node,
        edge,
        basis,
        src,
        dst,
        p + ".graph_modules.0.to_key_value",
        false,
        L,
        mask,
      ),
      C = fused.shape[1] / 2,
      H = 4,
      value = o.slice(fused, 1, 0, C),
      key = o.reshape(o.slice(fused, 1, C, C * 2), [L, L, H, -1]);
    const query = {};
    for (let d = 0; d < 2; d++)
      query[d] = o.einsum(
        "oc,nck->nok",
        this.w[p + `.graph_modules.0.to_query.weights.${d}`],
        node[d],
      );
    const q = o.reshape(o.cat([query[0], query[1]], 2), [L, H, -1]);
    let att = o.div(o.einsum("abhk,bhk->abh", key, q), Math.sqrt(C * 4));
    att = o.add(att, o.mul(o.sub(o.reshape(mask, [L, L, 1]), 1), 1e30));
    att = o.softmax(att, 0, true);
    const z = o.reshape(
      o.einsum(
        "abhck->bhck",
        o.mul(
          o.reshape(att, [L, L, H, 1, 1]),
          o.reshape(value, [L, L, H, C / H, 4]),
        ),
      ),
      [L, C, 4],
    );
    const next = {};
    for (let d = 0; d < 2; d++) {
      const v = o.slice(z, 2, d === 0 ? 0 : 1, d === 0 ? 1 : 4);
      next[d] = o.einsum(
        "oc,nck->nok",
        this.w[p + `.graph_modules.0.project.weights.${d}`],
        o.cat([v, node[d]], 1),
      );
    }
    const norms = [
        normLast(o, next[0], 2 ** -24),
        normLast(o, next[1], 2 ** -24),
      ],
      normed = {};
    for (let d = 0; d < 2; d++) {
      const channels = next[d].shape[1],
        weight = o.slice(
          this.w[p + ".graph_modules.1.group_norm.weight"],
          0,
          d * channels,
          (d + 1) * channels,
        ),
        bias = o.slice(
          this.w[p + ".graph_modules.1.group_norm.bias"],
          0,
          d * channels,
          (d + 1) * channels,
        );
      const n = o.reshape(
        o.relu(o.groupNorm(o.reshape(norms[d], [L, channels]), weight, bias)),
        [L, channels, 1],
      );
      normed[d] = o.mul(o.div(next[d], norms[d]), n);
    }
    return this.conv(
      normed,
      edge,
      basis,
      src,
      dst,
      p + ".graph_modules.2",
      true,
      L,
      mask,
    );
  }
  graphMask(CA, indices, topk) {
    const o = this.o,
      L = CA.shape[0];
    if (!topk)
      return o.tensor(
        [L, L],
        Array.from({ length: L * L }, (_, i) =>
          Number(indices[Math.floor(i / L)] !== indices[i % L]),
        ),
      );
    if (L <= 64) return o.constant([L, L], 1);
    const dist = cdist(o, CA),
      ix = o.tensor([L], indices);
    return o.kernel(
      [L, L],
      [dist, ix],
      `const row=Math.floor(i/${L}),col=i%${L},sep=Math.abs(a1[row]-a1[col])+(row===col?999.9:0),d=a0[i]+(row===col?999.9:0)+sep*1e-6;let rank=0;for(let j=0;j<${L};j++){const s=Math.abs(a1[row]-a1[j])+(row===j?999.9:0),v=a0[row*${L}+j]+(row===j?999.9:0)+s*1e-6;if(v<d||(v===d&&j<col))rank++;}o[i]=sep<32||rank<64?1:0;`,
      `let row=i/${L}u;let col=i%${L}u;let sep=abs(a1[row]-a1[col])+select(0.0,999.9,row==col);let d=a0[i]+select(0.0,999.9,row==col)+sep*1e-6;var rank=0u;for(var j=0u;j<${L}u;j++){let s=abs(a1[row]-a1[j])+select(0.0,999.9,row==j);let v=a0[row*${L}u+j]+select(0.0,999.9,row==j)+s*1e-6;if(v<d||(v==d&&j<col)){rank++;}}o[i]=select(0.0,1.0,sep<32.0||rank<64u);`,
    );
  }
  structure(msa, pair, R, T, xyz, state, indices, motif, p, topk) {
    const o = this.o,
      L = msa.shape[0];
    let node = o.cat([
      this.norm(msa, p + ".norm_msa"),
      this.norm(state, p + ".norm_state"),
    ]);
    node = this.norm(this.linear(node, p + ".embed_x"), p + ".norm_node");
    pair = this.norm(
      this.linear(this.norm(pair, p + ".norm_pair"), p + ".embed_e1"),
      p + ".norm_edge1",
    );
    const CA = o.reshape(o.slice(xyz, 1, 1, 2), [L, 3]),
      neighbor = o.tensor(
        [L, L, 1],
        Array.from({ length: L * L }, (_, i) => {
          const d = indices[i % L] - indices[Math.floor(i / L)];
          return Math.abs(d) <= 1 ? d : 0;
        }),
      );
    pair = this.norm(
      this.linear(
        o.cat([pair, rbf(o, cdist(o, CA)), neighbor]),
        p + ".embed_e2",
      ),
      p + ".norm_edge2",
    );
    const l1 = o.sub(xyz, o.reshape(CA, [L, 1, 3]));
    const shift = this.se3(
      { 0: o.reshape(node, [L, node.shape[1], 1]), 1: l1 },
      pair,
      CA,
      p + ".se3.se3",
      topk,
      indices,
    );
    state = o.reshape(shift[0], [L, -1]);
    const offset = o.mul(
        shift[1],
        o.tensor(
          [L, 1, 1],
          Array.from(motif, (x) =>
            this.c.freeze_track_motif ? 1 - Number(x) : 1,
          ),
        ),
      ),
      dT = o.div(o.reshape(o.slice(offset, 1, 0, 1), [L, 3]), 10),
      dR = quaternionRotation(
        o,
        o.div(o.reshape(o.slice(offset, 1, 1, 2), [L, 3]), 100),
      );
    return {
      R: o.einsum("nij,njk->nik", dR, R),
      T: o.add(T, dT),
      state,
      alpha: this.sc(msa, state, p + ".sc_predictor"),
    };
  }
  async forward(
    input,
    { signal, returnIntermediates = false, onProgress } = {},
  ) {
    const o = this.o,
      L = input.length,
      trace = {};
    const take = async (name, x) => {
      if (returnIntermediates) trace[name] = await o.read(x);
    };
    const get = (k, s) => o.tensor(s, input[k]);
    abort(signal);
    const seq = get("seq", [L, 22]),
      xyz = get("xyz", [L, 3, 3]),
      indices = input.idx,
      motif = input.motif_mask ?? new Uint8Array(L);
    let msa = o.add(
        this.linear(get("msa_latent", [L, 48]), "latent_emb.emb"),
        o.einsum("lk,kc->lc", seq, this.w["latent_emb.emb_q.weight"]),
      ),
      full = o.add(
        this.linear(get("msa_full", [L, 25]), "full_emb.emb"),
        o.einsum("lk,kc->lc", seq, this.w["full_emb.emb_q.weight"]),
      );
    let pair = o.add(
      o.reshape(
        o.einsum("lk,kc->lc", seq, this.w["latent_emb.emb_left.weight"]),
        [1, L, -1],
      ),
      o.reshape(
        o.einsum("lk,kc->lc", seq, this.w["latent_emb.emb_right.weight"]),
        [L, 1, -1],
      ),
    );
    const pos = o.tensor(
      [L, L],
      Array.from({ length: L * L }, (_, i) =>
        Math.max(
          0,
          Math.min(64, indices[i % L] - indices[Math.floor(i / L)] + 32),
        ),
      ),
    );
    pair = o.add(pair, o.gather(this.w["latent_emb.pos.emb.weight"], pos));
    let state = o.einsum(
      "lk,kc->lc",
      seq,
      this.w["latent_emb.emb_state.weight"],
    );
    await take("latent_emb.msa", msa);
    await take("latent_emb.pair", pair);
    await take("latent_emb.state", state);
    await take("full_emb", full);
    const prevState = this.norm(
        input.state_prev
          ? get("state_prev", state.shape)
          : o.constant(state.shape),
        "recycle.norm_state",
      ),
      prevMSA = this.norm(
        input.msa_prev ? get("msa_prev", msa.shape) : o.constant(msa.shape),
        "recycle.norm_msa",
      ),
      prevPair = this.norm(
        input.pair_prev ? get("pair_prev", pair.shape) : o.constant(pair.shape),
        "recycle.norm_pair",
      );
    const dist = rbf(o, cdist(o, cbeta(o, xyz))),
      left = o.expr(
        [L, L, prevState.shape[1]],
        [o.reshape(prevState, [L, 1, -1])],
        "$0",
      ),
      right = o.expr(
        [L, L, prevState.shape[1]],
        [o.reshape(prevState, [1, L, -1])],
        "$0",
      );
    msa = o.add(msa, prevMSA);
    pair = o.add(
      pair,
      o.add(
        prevPair,
        this.linear(o.cat([dist, left, right]), "recycle.proj_dist"),
      ),
    );
    state = o.add(state, prevState);
    await take("recycle.msa", msa);
    await take("recycle.pair", pair);
    await take("recycle.state", state);
    const t1d = get("t1d", [L, input.t1d.length / L]),
      t2d = get("t2d", [L, L, input.t2d.length / L / L]),
      alpha_t = get("alpha_t", [L, 30]),
      xyz_t = get("xyz_t", [L, 3, 3]);
    let templ = this.linear(
      o.cat([
        t2d,
        o.expr([L, L, t1d.shape[1]], [o.reshape(t1d, [L, 1, -1])], "$0"),
        o.expr([L, L, t1d.shape[1]], [o.reshape(t1d, [1, L, -1])], "$0"),
      ]),
      "templ_emb.emb",
    );
    const templDist = rbf(
      o,
      cdist(o, o.reshape(o.slice(xyz_t, 1, 1, 2), [L, 3])),
    );
    for (let j = 0; j < 2; j++) {
      templ = this.pairBlock(
        templ,
        templDist,
        `templ_emb.templ_stack.block.${j}`,
      );
      await take(`templ_emb.templ_stack.block.${j}`, templ);
    }
    templ = this.norm(templ, "templ_emb.templ_stack.norm");
    const tors = this.linear(
      o.relu(this.linear(o.cat([t1d, alpha_t]), "templ_emb.emb_t1d")),
      "templ_emb.proj_t1d",
    );
    // One template: pointwise attention softmax has exactly one key.
    state = o.add(
      state,
      this.linear(
        this.linear(tors, "templ_emb.attn_tor.to_v"),
        "templ_emb.attn_tor.to_out",
      ),
    );
    pair = o.add(
      pair,
      this.linear(
        this.linear(templ, "templ_emb.attn.to_v"),
        "templ_emb.attn.to_out",
      ),
    );
    await take("templ_emb.pair", pair);
    await take("templ_emb.state", state);
    let R,
      T = o.reshape(o.slice(xyz, 1, 1, 2), [L, 3]),
      centered = o.sub(xyz, o.reshape(T, [L, 1, 3]));
    R = o.tensor(
      [L, 3, 3],
      Array.from({ length: L * 9 }, (_, i) =>
        Number([0, 4, 8].includes(i % 9)),
      ),
    );
    state = this.linear(state, "simulator.proj_state");
    let alpha;
    const stages = [
      ...Array.from({ length: this.c.n_extra_block }, (_, j) => [
        "extra_block",
        j,
      ]),
      ...Array.from({ length: this.c.n_main_block }, (_, j) => [
        "main_block",
        j,
      ]),
      ...Array.from({ length: this.c.n_ref_block }, (_, j) => [
        "str_refiner",
        j,
      ]),
    ];
    for (let k = 0; k < stages.length; k++) {
      abort(signal);
      const [kind, j] = stages[k],
        ref = kind === "str_refiner",
        p = "simulator." + kind + (ref ? "" : "." + j),
        global = kind === "extra_block";
      if (ref && j === 0) state = this.linear(state, "simulator.proj_state2");
      const current = o.add(
        o.einsum("nij,naj->nai", R, centered),
        o.reshape(T, [L, 1, 3]),
      );
      let m = global ? full : msa;
      if (!ref) {
        const d = rbf(
          o,
          cdist(o, o.reshape(o.slice(current, 1, 1, 2), [L, 3])),
        );
        m = this.msaBlock(m, pair, d, state, p + ".msa2msa", global);
        pair = this.msa2pair(m, pair, p + ".msa2pair");
        pair = this.pairBlock(pair, d, p + ".pair2pair");
        if (global) full = m;
        else msa = m;
      }
      ({ R, T, state, alpha } = this.structure(
        m,
        pair,
        R,
        T,
        current,
        state,
        indices,
        motif,
        p + (ref ? "" : ".str2str"),
        ref,
      ));
      await take(`${p}.${j}.R`, R);
      await take(`${p}.${j}.T`, T);
      await take(`${p}.${j}.state`, state);
      await take(`${p}.${j}.msa`, m);
      await take(`${p}.${j}.pair`, pair);
      await take(`${p}.${j}.alpha`, alpha);
      onProgress?.({ phase: "denoise", block: k + 1, total: stages.length });
      await o.checkpoint([msa, full, pair, R, T, state, alpha, centered, xyz]);
    }
    const px0 = o.add(
        o.einsum("nij,naj->nai", R, centered),
        o.reshape(T, [L, 1, 3]),
      ),
      logits = this.linear(msa, "aa_pred.proj"),
      lddt = o.einsum(
        "nk,k->n",
        o.softmax(this.linear(state, "lddt_pred.proj")),
        o.tensor([50], this.constants.lddtBins),
      );
    await take("px0", px0);
    await take("alpha", alpha);
    await take("logits", logits);
    await take("plddt", lddt);
    const coords = backbone(
      o,
      px0,
      alpha,
      Array.from({ length: L }, (_, i) => {
        let k = 0;
        for (let j = 1; j < 22; j++)
          if (input.seq[i * 22 + j] > input.seq[i * 22 + k]) k = j;
        return k;
      }),
    );
    await take("backbone", coords);
    const result = {
      coords: await o.read(coords),
      px0: await o.read(px0),
      alpha: await o.read(alpha),
      logits: await o.read(logits),
      plddt: await o.read(lddt),
      intermediates: trace,
    };
    abort(signal);
    return result;
  }
}
