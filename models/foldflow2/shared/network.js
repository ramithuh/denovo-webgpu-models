import {
  normalize,
  rotation,
  update,
  points,
  inversePoints,
  backbone,
} from "./geometry.js";
import { abort } from "./loader.js";
export class FoldFlow2Network {
  constructor(o, w) {
    this.o = o;
    this.w = w;
    this.keep = [];
    this.trace = {};
  }
  lin(x, p) {
    return this.o.linear(x, this.w[p + ".weight"], this.w[p + ".bias"]);
  }
  norm(x, p) {
    return this.o.norm(x, this.w[p + ".weight"], this.w[p + ".bias"]);
  }
  async mark(name, x) {
    if (this.tracing) this.trace[name] = await this.o.read(x);
    return x;
  }
  async boundary(xs) {
    await this.o.checkpoint([...this.keep, ...xs]);
    abort(this.signal);
  }
  mlp(x, p) {
    const o = this.o;
    return this.lin(
      o.relu(this.lin(this.norm(x, p + ".0"), p + ".1")),
      p + ".3",
    );
  }
  async esm(L, tokens) {
    const o = this.o,
      N = L + 2,
      D = 1280,
      H = 20,
      dh = 64,
      p = "seq_encoder.esm";
    tokens = tokens ?? [0, ...Array(L).fill(32), 2];
    if (
      tokens.length !== N ||
      tokens[0] !== 0 ||
      tokens[N - 1] !== 2 ||
      Array.from(tokens).some(
        (t) => !Number.isInteger(t) || t < 0 || t > 32 || t === 1,
      )
    )
      throw Error("ESM tokens require BOS=0, EOS=2 and unpadded IDs in [0,32]");
    const masked = tokens.filter((x) => x === 32).length,
      den = Math.fround(1 - Math.fround(masked / N));
    let x = o.gather(this.w[p + ".embed_tokens.weight"], o.tensor([N], tokens));
    x = o.mul(
      o.mul(
        x,
        o.tensor(
          [N, 1],
          tokens.map((x) => (x === 32 ? 0 : 1)),
        ),
      ),
      0.88,
    );
    x = o.div(x, den);
    const reps = [o.slice(x, 0, 1, N - 1)],
      att = [];
    for (let b = 0; b < 33; b++) {
      const pre = `${p}.layers.${b}`,
        z = this.norm(x, pre + ".self_attn_layer_norm");
      let q = o.mul(
          o.reshape(this.lin(z, pre + ".self_attn.q_proj"), [N, H, dh]),
          0.125,
        ),
        k = o.reshape(this.lin(z, pre + ".self_attn.k_proj"), [N, H, dh]);
      const v = o.reshape(this.lin(z, pre + ".self_attn.v_proj"), [N, H, dh]);
      const rope = (a) =>
        o.kernel(
          a.shape,
          [a, this.w[pre + ".self_attn.rot_emb.inv_freq"]],
          `const r=Math.floor(i/${H * dh}),c=i%${dh},j=c%32,off=i-c,ang=Math.fround(r*a1[j]),co=Math.fround(Math.cos(ang)),si=Math.fround(Math.sin(ang));o[i]=Math.fround(a0[i]*co)+Math.fround((c<32?-a0[off+c+32]:a0[off+c-32])*si);`,
          `let r=i/${H * dh}u;let c=i%64u;let j=c%32u;let off=i-c;let ang=f32(r)*a1[j];let co=cos(ang);let si=sin(ang);var other=0.0;if(c<32u){other=-a0[off+c+32u];}else{other=a0[off+c-32u];}o[i]=a0[i]*co+other*si;`,
        );
      q = rope(q);
      k = rope(k);
      const a = o.softmax(o.einsum("qhd,khd->hqk", q, k));
      att.push(o.slice(o.slice(a, 1, 1, N - 1), 2, 1, N - 1));
      x = o.add(
        x,
        this.lin(
          o.reshape(o.einsum("hqk,khd->qhd", a, v), [N, D]),
          pre + ".self_attn.out_proj",
        ),
      );
      x = o.add(
        x,
        this.lin(
          this.gelu(
            this.lin(this.norm(x, pre + ".final_layer_norm"), pre + ".fc1"),
          ),
          pre + ".fc2",
        ),
      );
      if (b === 32) x = this.norm(x, p + ".emb_layer_norm_after");
      reps.push(o.slice(x, 0, 1, N - 1));
      await this.boundary([x, ...reps, ...att]);
      this.onProgress?.({ phase: "esm", layer: b + 1, total: 33 });
    }
    const concat = (xs, axis) => {
      while (xs.length > 1) {
        const next = [];
        for (let i = 0; i < xs.length; i += 6)
          next.push(
            xs.length - i === 1 ? xs[i] : o.cat(xs.slice(i, i + 6), axis),
          );
        xs = next;
      }
      return xs[0];
    };
    const single = concat(
      reps.map((x) => o.reshape(x, [L, 1, D])),
      1,
    );
    const pair = o.permute(
      concat(
        att.map((a) => o.reshape(a, [1, H, L, L])),
        0,
      ),
      [3, 2, 0, 1],
    );
    return [single, o.reshape(pair, [L, L, 660])];
  }
  gelu(x) {
    const o = this.o;
    if (o.fastGelu) return o.fastGelu(x);
    return o.kernel(
      x.shape,
      [x],
      `const f=Math.fround,z=a0[i];o[i]=f(z*.5)*f(1+rfErf(f(z/f(Math.SQRT2))));`,
      `let z=a0[i];o[i]=(z*0.5)*(1.0+rfErf(rfdiv(z,1.4142135623730951)));`,
    );
  }
  sequence(single, pair, idx) {
    const o = this.o,
      L = idx.length,
      p = "sequence_to_trunk_network";
    const coeff = o.softmax(this.w[p + ".esm_single_combine"]);
    let s = this.mlp(o.einsum("lnd,n->ld", single, coeff), p + ".single_mlp"),
      z = this.mlp(pair, p + ".pair_mlp");
    const rel = o.tensor(
      [L, L],
      idx.flatMap((i) =>
        idx.map((j) => Math.max(-32, Math.min(32, j - i)) + 33),
      ),
    );
    z = o.add(
      z,
      o.gather(
        this.w[p + ".pairwise_positional_embedding.embedding.weight"],
        rel,
      ),
    );
    return [s, z];
  }
  embedding(L, time, idx, sc, idxTensor = null) {
    const o = this.o,
      p = "bb_encoder.embedding_layer";
    const embed = (x, n) =>
      this.norm(
        this.lin(
          o.relu(
            this.lin(
              o.relu(this.lin(x, p + "." + n + ".0")),
              p + "." + n + ".2",
            ),
          ),
          p + "." + n + ".4",
        ),
        p + "." + n + ".5",
      );
    if (o.fastEmbedFeatures && idxTensor) {
      const [singleFeat, edgeFeat] = o.fastEmbedFeatures(idxTensor, time, sc, L);
      return [
        embed(singleFeat, "node_embedder"),
        embed(edgeFeat, "edge_embedder"),
      ];
    }
    const f = Math.fround,
      te = Array.from({ length: 32 }, (_, c) => {
        let a = f(
          f(f(time) * 10000) * f(Math.exp(f((c % 16) * f(-Math.log(10000) / 15)))),
        );
        return f(c < 16 ? Math.sin(a) : Math.cos(a));
      });
    const positional = (i, c) => {
      // PyTorch casts the Python scalar before multiplying the index tensor.
      const a = f(f(f(i) * f(Math.PI)) / f(2056 ** f((c % 16) / 16)));
      return f(c < 16 ? Math.sin(a) : Math.cos(a));
    };
    const single = o.tensor(
      [L, 65],
      idx.flatMap((i) => [
        ...te,
        0,
        ...Array.from({ length: 32 }, (_, c) => positional(i, c)),
      ]),
    );
    const pairBase = o.tensor(
      [L, L, 98],
      idx.flatMap((i) =>
        idx.flatMap((j) => [
          ...te,
          0,
          ...te,
          0,
          ...Array.from({ length: 32 }, (_, c) => positional(i - j, c)),
        ]),
      ),
    );
    const dist = o.kernel(
      [L, L, 22],
      [sc],
      `const r=Math.floor(i/22),a=Math.floor(r/${L})*3,b=r%${L}*3,k=i%22;let d=0;for(let j=0;j<3;j++)d+=Math.fround(a0[a+j]-a0[b+j])**2;d=Math.sqrt(d);const lo=Math.fround(.00001+k*(20-.00001)/21),hi=k===21?1e8:Math.fround(.00001+(k+1)*(20-.00001)/21);o[i]=d>lo&&d<hi?1:0;`,
      `let r=i/22u;let a=r/${L}u*3u;let b=r%${L}u*3u;let k=i%22u;var d=0.0;for(var j=0u;j<3u;j++){let x=a0[a+j]-a0[b+j];d+=x*x;}d=sqrt(d);let lo=0.00001+f32(k)*(20.0-0.00001)/21.0;var hi=1e8;if(k<21u){hi=0.00001+f32(k+1u)*(20.0-0.00001)/21.0;}o[i]=select(0.0,1.0,d>lo&&d<hi);`,
    );
    return [
      embed(single, "node_embedder"),
      embed(o.cat([pairBase, dist]), "edge_embedder"),
    ];
  }
  ipa(s, z, q, t, p) {
    const o = this.o,
      L = s.shape[0],
      H = 8,
      C = 256,
      P = 8,
      V = 12;
    const qr = o.reshape(this.lin(s, p + ".linear_q"), [L, H, C]),
      kv = o.reshape(this.lin(s, p + ".linear_kv"), [L, H, C * 2]);
    const k = o.slice(kv, 2, 0, C),
      v = o.slice(kv, 2, C, C * 2);
    const qp = points(o, this.lin(s, p + ".linear_q_points"), q, t, H, P),
      kvp = points(o, this.lin(s, p + ".linear_kv_points"), q, t, H, P + V),
      kp = o.slice(kvp, 2, 0, P),
      vp = o.slice(kvp, 2, P, P + V);
    let a = o.add(
      o.mul(o.einsum("qhc,khc->hqk", qr, k), Math.sqrt(1 / (3 * C))),
      o.mul(
        o.permute(this.lin(z, p + ".linear_b"), [2, 0, 1]),
        Math.sqrt(1 / 3),
      ),
    );
    const hw = this.w[p + ".head_weights"];
    const pa = o.kernel(
      [H, L, L],
      [qp, kp, hw],
      `const h=Math.floor(i/${L * L}),a=Math.floor(i/${L})%${L},b=i%${L};let sum=0;for(let p=0;p<8;p++){let d=0;for(let c=0;c<3;c++){let v=a0[((a*8+h)*8+p)*3+c]-a1[((b*8+h)*8+p)*3+c];d+=v*v;}sum+=d;}o[i]=-.5*sum*Math.log1p(Math.exp(a2[h]))*${Math.sqrt(1 / 108)};`,
      `let h=i/${L * L}u;let a=i/${L}u%${L}u;let b=i%${L}u;var sum=0.0;for(var p=0u;p<8u;p++){var d=0.0;for(var c=0u;c<3u;c++){let v=a0[((a*8u+h)*8u+p)*3u+c]-a1[((b*8u+h)*8u+p)*3u+c];d+=v*v;}sum+=d;}o[i]=-0.5*sum*log(1.0+exp(a2[h]))*${Math.sqrt(1 / 108)};`,
    );
    a = o.softmax(o.add(a, pa));
    const value = o.reshape(o.einsum("hqk,khc->qhc", a, v), [L, H * C]);
    const pt = inversePoints(o, o.einsum("hqk,khpc->qhpc", a, vp), q, t);
    const comps = [0, 1, 2].map((c) =>
      o.reshape(o.slice(pt, 3, c, c + 1), [L, H * V]),
    );
    const norms = o.kernel(
      [L, H * V],
      [pt],
      `const j=i*3;o[i]=Math.sqrt(a0[j]**2+a0[j+1]**2+a0[j+2]**2+1e-8);`,
      `let j=i*3u;o[i]=sqrt(a0[j]*a0[j]+a0[j+1u]*a0[j+1u]+a0[j+2u]*a0[j+2u]+1e-8);`,
    );
    const pair = o.reshape(
      o.einsum("hqk,qkc->qhc", a, this.lin(z, p + ".down_z")),
      [L, H * 32],
    );
    return this.lin(o.cat([value, ...comps, norms, pair]), p + ".linear_out");
  }
  transformer(x, p) {
    const o = this.o,
      L = x.shape[0],
      D = x.shape[1],
      H = 4,
      d = D / H;
    const w = this.w[p + ".self_attn.in_proj_weight"],
      b = this.w[p + ".self_attn.in_proj_bias"];
    const qkv = o.linear(x, w, b),
      pieces = [0, 1, 2].map((k) =>
        o.reshape(o.slice(qkv, 1, k * D, (k + 1) * D), [L, H, d]),
      );
    const a = o.softmax(
      o.einsum("qhc,khc->hqk", o.mul(pieces[0], 1 / Math.sqrt(d)), pieces[1]),
    );
    let y = this.norm(
      o.add(
        x,
        this.lin(
          o.reshape(o.einsum("hqk,khc->qhc", a, pieces[2]), [L, D]),
          p + ".self_attn.out_proj",
        ),
      ),
      p + ".norm1",
    );
    return this.norm(
      o.add(y, this.lin(o.relu(this.lin(y, p + ".linear1")), p + ".linear2")),
      p + ".norm2",
    );
  }
  async structure(s, z, q, t, p) {
    const o = this.o,
      L = s.shape[0],
      initial = s;
    t = o.mul(t, 0.1);
    q = normalize(o, q);
    for (let b = 0; b < 2; b++) {
      const pre = p + ".trunk";
      s = this.norm(
        o.add(s, this.ipa(s, z, q, t, `${pre}.ipa_${b}`)),
        `${pre}.ipa_ln_${b}`,
      );
      let tf = o.cat([s, this.lin(initial, `${pre}.skip_embed_${b}`)]);
      for (let j = 0; j < 2; j++)
        tf = this.transformer(tf, `${pre}.seq_tfmr_${b}.layers.${j}`);
      s = o.add(s, this.lin(tf, `${pre}.post_tfmr_${b}`));
      const n = `${pre}.node_transition_${b}`;
      s = this.norm(
        o.add(
          s,
          this.lin(
            o.relu(
              this.lin(o.relu(this.lin(s, n + ".linear_1")), n + ".linear_2"),
            ),
            n + ".linear_3",
          ),
        ),
        n + ".ln",
      );
      [q, t] = update(o, q, t, this.lin(s, `${pre}.bb_update_${b}.linear`));
      if (b < 1 || p === "bb_encoder") {
        const e = `${pre}.edge_transition_${b}`,
          node = this.lin(s, e + ".initial_embed"),
          left = o.expr([L, L, 128], [o.reshape(node, [L, 1, 128])], "$0"),
          right = o.expr([L, L, 128], [o.reshape(node, [1, L, 128])], "$0"),
          zz = o.cat([z, left, right]);
        z = this.norm(
          this.lin(
            o.add(
              zz,
              o.relu(
                this.lin(o.relu(this.lin(zz, e + ".trunk.0")), e + ".trunk.2"),
              ),
            ),
            e + ".final_layer",
          ),
          e + ".layer_norm",
        );
      }
      await this.mark(`${p}.${b}.s`, s);
      await this.mark(`${p}.${b}.q`, q);
      await this.boundary([s, z, q, t, initial]);
    }
    const a = p + ".torsion_pred",
      psi = normalize(
        o,
        this.lin(
          o.add(
            s,
            this.lin(o.relu(this.lin(s, a + ".linear_1")), a + ".linear_2"),
          ),
          a + ".linear_final",
        ),
      );
    return [s, z, q, o.div(t, 0.1), psi];
  }
  triMul(z, p, incoming) {
    const o = this.o;
    z = this.norm(z, p + ".layer_norm_in");
    const a = o.mul(
        this.lin(z, p + ".linear_a_p"),
        o.sigmoid(this.lin(z, p + ".linear_a_g")),
      ),
      b = o.mul(
        this.lin(z, p + ".linear_b_p"),
        o.sigmoid(this.lin(z, p + ".linear_b_g")),
      );
    return o.mul(
      this.lin(
        this.norm(
          o.einsum(incoming ? "kic,kjc->ijc" : "ikc,jkc->ijc", a, b),
          p + ".layer_norm_out",
        ),
        p + ".linear_z",
      ),
      o.sigmoid(this.lin(z, p + ".linear_g")),
    );
  }
  triAttention(z, p, end) {
    const o = this.o,
      L = z.shape[0],
      H = 4,
      d = 32;
    if (end) z = o.permute(z, [1, 0, 2]);
    z = this.norm(z, p + ".layer_norm");
    const q = o.mul(
        o.reshape(this.lin(z, p + ".mha.linear_q"), [L, L, H, d]),
        1 / Math.sqrt(d),
      ),
      k = o.reshape(this.lin(z, p + ".mha.linear_k"), [L, L, H, d]),
      v = o.reshape(this.lin(z, p + ".mha.linear_v"), [L, L, H, d]);
    const bias = o.reshape(o.permute(this.lin(z, p + ".linear"), [2, 0, 1]), [
      1,
      H,
      L,
      L,
    ]);
    const a = o.softmax(o.add(o.einsum("iqhc,ikhc->ihqk", q, k), bias));
    let out = this.lin(
      o.mul(
        o.reshape(o.einsum("ihqk,ikhc->iqhc", a, v), [L, L, H * d]),
        o.sigmoid(this.lin(z, p + ".mha.linear_g")),
      ),
      p + ".mha.linear_o",
    );
    return end ? o.permute(out, [1, 0, 2]) : out;
  }
  async trunk(s, z) {
    const o = this.o,
      L = s.shape[0];
    for (let b = 0; b < 2; b++) {
      const p = `trunk_network.blocks.${b}`,
        bias = this.lin(
          this.norm(z, p + ".pair_to_sequence.layernorm"),
          p + ".pair_to_sequence.linear",
        ),
        x = this.norm(s, p + ".layernorm_1"),
        qkv = o.reshape(this.lin(x, p + ".seq_attention.proj"), [L, 8, 96]);
      const q = o.mul(o.slice(qkv, 2, 0, 32), 1 / Math.sqrt(32)),
        k = o.slice(qkv, 2, 32, 64),
        v = o.slice(qkv, 2, 64, 96),
        a = o.softmax(
          o.add(o.einsum("qhc,khc->hqk", q, k), o.permute(bias, [2, 0, 1])),
        );
      s = o.add(
        s,
        this.lin(
          o.mul(
            o.reshape(o.einsum("hqk,khc->qhc", a, v), [L, 256]),
            o.sigmoid(this.lin(x, p + ".seq_attention.g_proj")),
          ),
          p + ".seq_attention.o_proj",
        ),
      );
      s = o.add(s, this.mlp(s, p + ".mlp_seq.mlp"));
      const stp = this.lin(
          this.norm(s, p + ".sequence_to_pair.layernorm"),
          p + ".sequence_to_pair.proj",
        ),
        qa = o.reshape(o.slice(stp, 1, 0, 64), [1, L, 64]),
        kb = o.reshape(o.slice(stp, 1, 64, 128), [L, 1, 64]);
      z = o.add(
        z,
        this.lin(
          o.cat([o.mul(qa, kb), o.sub(qa, kb)]),
          p + ".sequence_to_pair.o_proj",
        ),
      );
      z = o.add(z, this.triMul(z, p + ".tri_mul_out", false));
      z = o.add(z, this.triMul(z, p + ".tri_mul_in", true));
      z = o.add(z, this.triAttention(z, p + ".tri_att_start", false));
      z = o.add(z, this.triAttention(z, p + ".tri_att_end", true));
      z = o.add(z, this.mlp(z, p + ".mlp_pair.mlp"));
      await this.mark(`trunk.${b}.s`, s);
      await this.mark(`trunk.${b}.z`, z);
      await this.boundary([s, z]);
    }
    return [s, z];
  }
  async forward(frames, time, sc, sequence, idx, idxTensor = null) {
    const o = this.o,
      L = idx.length;
    let q = o.slice(frames, 1, 0, 4),
      t = o.slice(frames, 1, 4, 7);
    const [s0, z0] = this.embedding(L, time, idx, sc, idxTensor);
    this.keep.push(s0, z0, frames, sc, idxTensor, ...sequence);
    let [s, z, nq, nt] = await this.structure(s0, z0, q, t, "bb_encoder");
    this.keep.push(nq, nt);
    const p = "combiner_network";
    s = this.norm(
      o.cat([
        this.lin(s, p + ".projections_single.bb"),
        this.lin(sequence[0], p + ".projections_single.seq"),
      ]),
      p + ".layer_norm_single",
    );
    z = this.norm(
      o.cat([
        this.lin(z, p + ".projections_pair.bb"),
        this.lin(sequence[1], p + ".projections_pair.seq"),
      ]),
      p + ".layer_norm_pair",
    );
    [s, z] = await this.trunk(s, z);
    const d = "trunk_to_decoder_network";
    s = o.mul(
      o.add(
        this.norm(this.lin(s, d + ".lin_single"), d + ".single_layernorm"),
        s0,
      ),
      0.5,
    );
    z = o.mul(
      o.add(this.norm(this.lin(z, d + ".lin_pair"), d + ".pair_layernorm"), z0),
      0.5,
    );
    let psi;
    [s, z, nq, nt, psi] = await this.structure(s, z, nq, nt, "bb_decoder");
    return {
      rigids: o.cat([nq, nt]),
      q: nq,
      t: nt,
      psi,
      atoms: backbone(o, nq, nt, psi),
    };
  }
}
