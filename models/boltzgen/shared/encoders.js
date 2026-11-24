export class AtomModules {
  constructor(ops, layers) {
    this.o = ops;
    this.layers = layers;
  }
  encoder(f, p, s, z) {
    const o = this.o;
    return o.tidy(() => {
      const A = f.ref_pos.shape[1],
        N = f.atom_to_token.shape[2],
        K = A / 32;
      const feats = o.cat([
        f.ref_pos,
        o.unsqueeze(f.ref_charge, -1),
        f.ref_element,
        o.reshape(f.ref_atom_name_chars, [1, A, 256]),
      ]);
      let c = o.linear(feats, p + ".embed_atom_features");
      const q = c;
      const d = o.sub(
        o.reshape(o.keys(f.ref_pos), [K, 1, 128, 3]),
        o.reshape(f.ref_pos, [K, 32, 1, 3]),
      );
      const dist = o.div(1, o.add(1, o.sum(o.mul(d, d), -1, true)));
      const km = o.reshape(o.keys(o.unsqueeze(f.atom_pad_mask, -1)), [
        K,
        1,
        128,
        1,
      ]);
      const qm = o.reshape(f.atom_pad_mask, [K, 32, 1, 1]);
      const uid = o.point(
        "eq",
        o.reshape(f.ref_space_uid, [K, 32, 1, 1]),
        o.reshape(o.keys(o.unsqueeze(f.ref_space_uid, -1)), [K, 1, 128, 1]),
      );
      const valid = o.mul(o.mul(km, qm), uid);
      let pair = o.mul(o.linear(d, p + ".embed_atompair_ref_pos"), valid);
      pair = o.add(
        pair,
        o.mul(o.linear(dist, p + ".embed_atompair_ref_dist"), valid),
      );
      pair = o.add(
        pair,
        o.mul(o.linear(valid, p + ".embed_atompair_mask"), valid),
      );
      if (s) {
        c = o.add(c, o.mm(f.atom_to_token, o.seq(s, p + ".s_to_c_trans")));
        const zp = o.permute(o.seq(z, p + ".z_to_p_trans"), [0, 3, 1, 2]);
        const aq = o.reshape(f.atom_to_token, [K, 1, 32, N]),
          ak = o.reshape(o.keys(f.atom_to_token), [K, 1, 128, N]);
        pair = o.add(
          pair,
          o.permute(o.mm(o.mm(aq, zp), o.transpose(ak)), [0, 2, 3, 1]),
        );
      }
      pair = o.add(
        pair,
        o.linear(
          o.relu(o.reshape(c, [K, 32, 1, 128])),
          p + ".c_to_p_trans_q.1",
        ),
      );
      pair = o.add(
        pair,
        o.linear(
          o.relu(o.reshape(o.keys(c), [K, 1, 128, 128])),
          p + ".c_to_p_trans_k.1",
        ),
      );
      let mlp = pair;
      for (const i of [1, 3, 5]) mlp = o.linear(o.relu(mlp), `${p}.p_mlp.${i}`);
      pair = o.add(pair, mlp);
      return { q, c, p: pair };
    });
  }
  attentionEncoder(f, q, c, bias, p, r, cache = null) {
    const o = this.o;
    return o.tidy(() => {
      if (r) q = o.add(q, o.linear(r, p + ".r_to_q_trans"));
      q = this.layers.atomTransformer(
        q,
        c,
        bias,
        f.atom_pad_mask,
        p + ".atom_encoder",
        cache?.atomEncLayers ?? null,
      );
      const qa = o.relu(o.linear(q, p + ".atom_to_token_trans.0"));
      const meanT =
        cache?.atomMeanT ??
        o.transpose(
          o.div(
            f.atom_to_token,
            o.add(o.sum(f.atom_to_token, 1, true), 1e-6),
          ),
        );
      return { a: o.mm(meanT, qa), q, c };
    });
  }
  input(f) {
    const o = this.o,
      p = "input_embedder";
    return o.tidy(() => {
      const { q, c, p: pair } = this.encoder(f, p + ".atom_encoder");
      const { a } = this.attentionEncoder(
        f,
        q,
        c,
        o.seq(pair, p + ".atom_enc_proj_z"),
        p + ".atom_attention_encoder",
      );
      let s = o.add(
        o.add(a, o.linear(f.res_type, p + ".res_type_encoding")),
        o.linear(
          o.cat([f.profile, o.unsqueeze(f.deletion_mean, -1)]),
          p + ".msa_profile_encoding",
        ),
      );
      for (const [name, feature] of [
        ["method", "method_feature"],
        ["modified", "modified"],
        ["mol_type", "mol_type"],
        ["design_mask", "design_mask"],
        ["binding_specification", "binding_type"],
      ])
        s = o.add(s, o.embed(f[feature], `${p}.${name}_conditioning_init`));
      s = o.add(
        s,
        o.linear(
          o.unsqueeze(o.point("clamp", f.cyclic, null, 1), -1),
          p + ".cyclic_conditioning_init",
        ),
      );
      s = o.add(s, o.embed(f.ss_type, p + ".ss_specification_init"));
      return s;
    });
  }
  conditioning(f, s, z, relative) {
    const o = this.o,
      p = "diffusion_conditioning";
    return o.tidy(() => {
      const pc = p + ".pairwise_conditioner";
      z = o.seq(o.cat([z, relative]), pc + ".dim_pairwise_init_proj");
      for (let i = 0; i < 2; i++)
        z = o.add(z, this.layers.transition(z, `${pc}.transitions.${i}`));
      const { q, c, p: pair } = this.encoder(f, p + ".atom_encoder", s, z);
      const biases = (name, t, n) =>
        o.cat(
          Array.from({ length: n }, (_, i) => o.seq(t, `${p}.${name}.${i}`)),
        );
      return {
        q,
        c,
        atom_enc_bias: biases("atom_enc_proj_z", pair, 3),
        atom_dec_bias: biases("atom_dec_proj_z", pair, 3),
        token_trans_bias: biases("token_trans_proj_z", z, 24),
      };
    });
  }
  decoder(f, a, q, c, bias, p, cache = null) {
    const o = this.o;
    return o.tidy(() => {
      q = o.add(q, o.mm(f.atom_to_token, o.linear(a, p + ".a_to_q_trans")));
      q = this.layers.atomTransformer(
        q,
        c,
        bias,
        f.atom_pad_mask,
        p + ".atom_decoder",
        cache?.atomDecLayers ?? null,
      );
      return o.seq(q, p + ".atom_feat_to_atom_pos_update");
    });
  }
}
