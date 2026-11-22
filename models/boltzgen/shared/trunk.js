export class Trunk {
  constructor(ops, layers) {
    this.o = ops;
    this.layers = layers;
  }
  distance(f, z, relative, pairMask) {
    const o = this.o,
      p = "token_distance_module";
    return o.tidy(() => {
      const N = z.shape[1],
        d = f.bb_dist;
      const offset = o.weight(
        p + ".token_distance_encoder.distance_gaussian_smearing.offset",
      );
      const delta = o.sub(d, offset),
        gauss = o.point(
          "exp",
          o.mul(o.mul(delta, delta), -0.5 / Math.fround(2 / 31) ** 2),
        );
      const rel = o.expand(o.unsqueeze(relative, 1), [1, 4, N, N, 128]);
      let dist = this.layers.transition(
        o.cat([gauss, d, rel]),
        p + ".token_distance_encoder.distance_token_bias_trans",
      );
      dist = o.reshape(o.permute(dist, [0, 2, 3, 4, 1]), [1, N, N, 512]);
      const a = o.linear(
        o.mul(
          o.cat([f.distance_bins, dist]),
          o.unsqueeze(f.token_distance_mask, -1),
        ),
        p + ".a_proj",
      );
      let v = o.add(o.linear(o.norm(z, p + ".z_norm"), p + ".z_proj"), a),
        pair = v;
      for (let i = 0; i < 2; i++)
        pair = this.layers.pair(pair, pairMask, `${p}.pairformer.layers.${i}`);
      v = o.norm(o.add(v, pair), p + ".v_norm");
      return o.linear(o.relu(v), p + ".u_proj");
    });
  }
  template(f, z, pairMask) {
    const o = this.o,
      p = "template_module";
    return o.tidy(() => {
      const [B, T, N] = f.template_mask.shape;
      const a = o.linear(f.template_features, p + ".a_proj");
      const base = o.reshape(
        o.add(
          o.unsqueeze(o.linear(o.norm(z, p + ".z_norm"), p + ".z_proj"), 1),
          a,
        ),
        [B * T, N, N, 64],
      );
      const pm = o.reshape(o.expand(o.unsqueeze(pairMask, 1), [B, T, N, N]), [
        B * T,
        N,
        N,
      ]);
      let v = base;
      for (let i = 0; i < 2; i++)
        v = this.layers.pair(v, pm, `${p}.pairformer.layers.${i}`);
      v = o.reshape(o.norm(o.add(base, v), p + ".v_norm"), [B, T, N, N, 64]);
      const mask = o.sum(f.template_mask, 2, false, "any");
      const count = o.point("clamp", o.sum(mask, 1), 1, null);
      const u = o.div(
        o.sum(o.mul(v, o.reshape(mask, [B, T, 1, 1, 1])), 1),
        o.reshape(count, [B, 1, 1, 1]),
      );
      return o.linear(o.relu(u), p + ".u_proj");
    });
  }
  pairAverage(m, z, mask, p) {
    const o = this.o;
    return o.tidy(() => {
      const [B, S, N] = m.shape;
      m = o.norm(m, p + ".norm_m");
      z = o.norm(z, p + ".norm_z");
      const v = o.permute(
        o.reshape(o.linear(m, p + ".proj_m"), [B, S, N, 8, 32]),
        [0, 3, 1, 2, 4],
      );
      const bias = o.add(
        o.permute(o.linear(z, p + ".proj_z"), [0, 3, 1, 2]),
        o.mul(o.sub(1, o.unsqueeze(mask, 1)), -1e6),
      );
      const y = o.reshape(
        o.permute(o.mm(o.unsqueeze(o.softmax(bias), 2), v), [0, 2, 3, 1, 4]),
        [B, S, N, 256],
      );
      return o.linear(
        o.mul(y, o.sigmoid(o.linear(m, p + ".proj_g"))),
        p + ".proj_o",
      );
    });
  }
  outer(m, mask, p) {
    const o = this.o;
    return o.tidy(() => {
      const [B, S, N] = m.shape;
      m = o.norm(m, p + ".norm");
      const a = o.mul(o.linear(m, p + ".proj_a"), o.unsqueeze(mask, -1)),
        b = o.mul(o.linear(m, p + ".proj_b"), o.unsqueeze(mask, -1));
      let prod = o.mm(
        o.reshape(o.permute(a, [0, 2, 3, 1]), [B, N * 32, S]),
        o.reshape(b, [B, S, N * 32]),
      );
      prod = o.reshape(
        o.permute(o.reshape(prod, [B, N, 32, N, 32]), [0, 1, 3, 2, 4]),
        [B, N, N, 1024],
      );
      const count = o.point("clamp", o.mm(o.transpose(mask), mask), 1, null);
      return o.linear(o.div(prod, o.unsqueeze(count, -1)), p + ".proj_o");
    });
  }
  msa(f, z, emb, pairMask) {
    const o = this.o,
      p = "msa_module";
    return o.tidy(() => {
      let m = o.linear(
        o.cat([
          o.onehot(f.msa, 33),
          o.unsqueeze(f.has_deletion, -1),
          o.unsqueeze(f.deletion_value, -1),
          o.unsqueeze(f.msa_paired, -1),
        ]),
        p + ".msa_proj",
      );
      m = o.add(m, o.unsqueeze(o.linear(emb, p + ".s_proj"), 1));
      for (let i = 0; i < 4; i++) {
        const l = `${p}.layers.${i}`;
        m = o.add(
          m,
          this.pairAverage(m, z, pairMask, l + ".pair_weighted_averaging"),
        );
        m = o.add(m, this.layers.transition(m, l + ".msa_transition"));
        z = o.add(z, this.outer(m, f.msa_mask, l + ".outer_product_mean"));
        z = this.layers.pair(z, pairMask, l + ".pairformer_layer");
      }
      return z;
    });
  }
  contact(f) {
    const o = this.o,
      p = "contact_conditioning";
    return o.tidy(() => {
      const N = f.contact_threshold.shape[1],
        cut = o.div(o.sub(f.contact_threshold, 4), 16);
      const fourier = o.reshape(
        o.point(
          "cos",
          o.mul(
            o.linear(o.reshape(cut, [-1, 1]), p + ".fourier_embedding.proj"),
            2 * Math.PI,
          ),
        ),
        [1, N, N, 128],
      );
      const cc = f.contact_conditioning;
      let x = o.linear(
        o.cat([o.slice(cc, -1, 2, 5), o.unsqueeze(cut, -1), fourier]),
        p + ".encoder",
      );
      x = o.mul(x, o.sub(1, o.sum(o.slice(cc, -1, 0, 2), -1, true)));
      x = o.add(
        x,
        o.mul(o.weight(p + ".encoding_unspecified"), o.slice(cc, -1, 0, 1)),
      );
      return o.add(
        x,
        o.mul(o.weight(p + ".encoding_unselected"), o.slice(cc, -1, 1, 2)),
      );
    });
  }
}
