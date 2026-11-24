// Neural modules translated from the pinned BoltzGen architecture.
export class Layers {
  constructor(ops) {
    this.o = ops;
  }
  transition(x, p) {
    const o = this.o;
    return o.tidy(() => {
      x = o.norm(x, p + ".norm");
      return o.linear(
        o.mul(o.silu(o.linear(x, p + ".fc1")), o.linear(x, p + ".fc2")),
        p + ".fc3",
      );
    });
  }
  adalnStats(s, p) {
    const o = this.o;
    return o.tidy(() => {
      const sn = o.norm(s, p + ".s_norm");
      return {
        scale: o.sigmoid(o.linear(sn, p + ".s_scale")),
        bias: o.linear(sn, p + ".s_bias"),
      };
    });
  }
  adaln(a, s, p, cached = null) {
    const o = this.o;
    if (cached) {
      return o.add(o.mul(cached.scale, o.norm(a)), cached.bias);
    }
    s = o.norm(s, p + ".s_norm");
    return o.add(
      o.mul(o.sigmoid(o.linear(s, p + ".s_scale")), o.norm(a)),
      o.linear(s, p + ".s_bias"),
    );
  }
  conditionedTransition(a, s, p, cached = null) {
    const o = this.o;
    return o.tidy(() => {
      a = this.adaln(a, s, p + ".adaln", cached?.adaln2);
      const g = o.linear(a, p + ".swish_gate.0"),
        d = g.shape.at(-1) / 2;
      const b = o.mul(
        o.mul(o.slice(g, -1, 0, d), o.silu(o.slice(g, -1, d, 2 * d))),
        o.linear(a, p + ".a_to_b"),
      );
      const outGate =
        cached?.transGate ?? o.sigmoid(o.linear(s, p + ".output_projection.0"));
      return o.mul(outGate, o.linear(b, p + ".b_to_a"));
    });
  }
  attention(s, z, mask, p, heads, kIn = s, preAttnMask = null) {
    const o = this.o;
    return o.tidy(() => {
      const [B, N, C] = s.shape,
        M = kIn.shape[1],
        D = C / heads;
      let q = o.reshape(o.linear(s, p + ".proj_q"), [B, N, heads, D]),
        k = o.reshape(o.linear(kIn, p + ".proj_k"), [B, M, heads, D]);
      if (o.weight(p + ".q_norm.weight", true)) {
        q = o.norm(q, p + ".q_norm");
        k = o.norm(k, p + ".k_norm");
      }
      q = o.transpose(q, 1, 2);
      k = o.transpose(k, 1, 2);
      const v = o.transpose(
        o.reshape(o.linear(kIn, p + ".proj_v"), [B, M, heads, D]),
        1,
        2,
      );
      let attnMask = preAttnMask;
      if (!attnMask) {
        const bias = o.permute(
          o.weight(p + ".proj_z.0.weight", true) ? o.seq(z, p + ".proj_z") : z,
          [0, 3, 1, 2],
        );
        attnMask = o.add(
          o.mul(o.sub(1, o.reshape(mask, [B, 1, 1, M])), -1e6),
          bias,
        );
      }
      // PyTorch MATH SDPA scales both operands, improving range and matching rounding.
      const logits = o.add(
        o.mm(o.mul(q, D ** -0.25), o.transpose(o.mul(k, D ** -0.25))),
        attnMask,
      );
      const out = o.reshape(o.transpose(o.mm(o.softmax(logits), v), 1, 2), [
        B,
        N,
        C,
      ]);
      return o.linear(
        o.mul(out, o.sigmoid(o.linear(s, p + ".proj_g"))),
        p + ".proj_o",
      );
    });
  }
  diffusionLayer(a, s, bias, mask, p, heads, windowed = false, cached = null) {
    const o = this.o;
    return o.tidy(() => {
      const b = this.adaln(a, s, p + ".adaln", cached?.adaln1),
        k = windowed ? o.keys(b) : b,
        km =
          cached?.km ??
          (windowed
            ? o.reshape(o.keys(o.unsqueeze(mask, -1)), [a.shape[0], 128])
            : mask);
      const attnGate =
        cached?.attnGate ?? o.sigmoid(o.linear(s, p + ".output_projection_linear"));
      a = o.add(
        a,
        o.mul(
          attnGate,
          this.attention(
            b,
            bias,
            km,
            p + ".pair_bias_attn",
            heads,
            k,
            cached?.attnMask ?? null,
          ),
        ),
      );
      a = o.add(a, this.conditionedTransition(a, s, p + ".transition", cached));
      return o.weight(p + ".post_lnorm.weight", true)
        ? o.norm(a, p + ".post_lnorm")
        : a;
    });
  }
  prepareAtomTransformerCache(c, bias, mask, p) {
    const o = this.o;
    return o.tidy(() => {
      const N = c.shape[1],
        K = N / 32;
      const cWin = o.reshape(c, [K, 32, 128]);
      const maskWin = o.reshape(mask, [K, 32]);
      const km = o.reshape(o.keys(o.unsqueeze(maskWin, -1)), [K, 128]);
      const biasWin = o.reshape(bias, [K, 32, 128, 12]);
      return Array.from({ length: 3 }, (_, i) => {
        const lp = `${p}.diffusion_transformer.layers.${i}`;
        const bSlice = o.permute(o.slice(biasWin, -1, i * 4, i * 4 + 4), [
          0, 3, 1, 2,
        ]);
        const attnMask = o.add(
          o.mul(o.sub(1, o.reshape(km, [K, 1, 1, 128])), -1e6),
          bSlice,
        );
        return {
          km,
          attnMask,
          adaln1: this.adalnStats(cWin, lp + ".adaln"),
          attnGate: o.sigmoid(o.linear(cWin, lp + ".output_projection_linear")),
          adaln2: this.adalnStats(cWin, lp + ".transition.adaln"),
          transGate: o.sigmoid(
            o.linear(cWin, lp + ".transition.output_projection.0"),
          ),
        };
      });
    });
  }
  atomTransformer(q, c, bias, mask, p, layerCaches = null) {
    const o = this.o;
    return o.tidy(() => {
      const N = q.shape[1],
        K = N / 32;
      q = o.reshape(q, [K, 32, 128]);
      c = o.reshape(c, [K, 32, 128]);
      mask = o.reshape(mask, [K, 32]);
      bias = o.reshape(bias, [K, 32, 128, 12]);
      for (let i = 0; i < 3; i++)
        q = this.diffusionLayer(
          q,
          c,
          o.slice(bias, -1, i * 4, i * 4 + 4),
          mask,
          `${p}.diffusion_transformer.layers.${i}`,
          4,
          true,
          layerCaches?.[i] ?? null,
        );
      return o.reshape(q, [1, N, 128]);
    });
  }
  triangleMult(z, mask, p, incoming = false) {
    const o = this.o;
    return o.tidy(() => {
      const [B, N, , D] = z.shape,
        x = o.norm(z, p + ".norm_in");
      let ab = o.mul(
        o.mul(o.linear(x, p + ".p_in"), o.sigmoid(o.linear(x, p + ".g_in"))),
        o.unsqueeze(mask, -1),
      );
      let a = o.slice(ab, -1, 0, D),
        b = o.slice(ab, -1, D, 2 * D);
      a = o.contiguous(o.permute(a, incoming ? [0, 3, 2, 1] : [0, 3, 1, 2]));
      b = o.contiguous(o.permute(b, incoming ? [0, 3, 1, 2] : [0, 3, 2, 1]));
      const y = o.contiguous(o.permute(o.mm(a, b), [0, 2, 3, 1]));
      return o.mul(
        o.linear(o.norm(y, p + ".norm_out"), p + ".p_out"),
        o.sigmoid(o.linear(x, p + ".g_out")),
      );
    });
  }
  triangleAttention(z, mask, p, ending = false) {
    const o = this.o;
    return o.tidy(() => {
      if (ending) {
        z = o.transpose(z, 1, 2);
        mask = o.transpose(mask, 1, 2);
      }
      const [B, N, , C] = z.shape,
        H = o.weight(p + ".linear.weight").shape[0],
        D = o.weight(p + ".mha.linear_q.weight").shape[0] / H;
      const x = o.norm(z, p + ".layer_norm"),
        bias = o.unsqueeze(
          o.permute(o.linear(x, p + ".linear"), [0, 3, 1, 2]),
          1,
        );
      const project = (name) =>
        o.permute(
          o.reshape(o.linear(x, p + ".mha." + name), [B, N, N, H, D]),
          [0, 1, 3, 2, 4],
        );
      const q = o.div(project("linear_q"), Math.sqrt(D)),
        k = project("linear_k"),
        v = project("linear_v");
      let logits = o.mm(q, o.transpose(k));
      logits = o.add(
        logits,
        o.mul(o.sub(o.reshape(mask, [B, N, 1, 1, N]), 1), 1e9),
      );
      logits = o.add(logits, bias);
      let y = o.reshape(
        o.permute(o.mm(o.softmax(logits), v), [0, 1, 3, 2, 4]),
        [B, N, N, H * D],
      );
      y = o.linear(
        o.mul(y, o.sigmoid(o.linear(x, p + ".mha.linear_g"))),
        p + ".mha.linear_o",
      );
      return ending ? o.transpose(y, 1, 2) : y;
    });
  }
  pair(z, mask, p) {
    const o = this.o;
    return o.tidy(() => {
      z = o.add(z, this.triangleMult(z, mask, p + ".tri_mul_out"));
      z = o.add(z, this.triangleMult(z, mask, p + ".tri_mul_in", true));
      z = o.add(z, this.triangleAttention(z, mask, p + ".tri_att_start"));
      z = o.add(z, this.triangleAttention(z, mask, p + ".tri_att_end", true));
      return o.add(z, this.transition(z, p + ".transition_z"));
    });
  }
  pairformer(s, z, mask, pairMask, p) {
    const o = this.o;
    return o.tidy(() => {
      z = this.pair(z, pairMask, p);
      const sn = o.norm(s, p + ".pre_norm_s");
      s = o.add(s, this.attention(sn, z, mask, p + ".attention", 16));
      s = o.add(s, this.transition(s, p + ".transition_s"));
      if (o.weight(p + ".s_post_norm.weight", true))
        s = o.norm(s, p + ".s_post_norm");
      return { s, z };
    });
  }
}
