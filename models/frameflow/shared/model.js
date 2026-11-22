/** The handwritten dynamic FrameFlow network. B=1; L and T are runtime inputs. */
export function forward(o, input, cfg, progress) {
  const L = input.length,
    C = cfg.node_embed_size,
    E = cfg.edge_embed_size,
    a = cfg.ipa;
  const mask = o.tensor(input.res_mask, [L]),
    diffuse = o.tensor(input.diffuse_mask, [L]);
  const pos = o.tensor(input.res_idx, [L]);
  const trans = o.tensor(input.trans_t, [L, 3]),
    rot = o.tensor(input.rotmats_t, [L, 9]);
  const sc = o.tensor(input.trans_sc ?? new Float32Array(L * 3), [L, 3]);
  const P = cfg.node_features.c_pos_emb,
    D = cfg.node_features.c_timestep_emb;
  let s = o.linear(
    o.nodeFeatures(pos, mask, diffuse, P, D, input.so3_t, input.r3_t),
    "node_feature_net.linear",
  );
  const F = cfg.edge_features.feat_dim,
    B = cfg.edge_features.num_bins;
  const sp = o.linear(s, "edge_feature_net.linear_s_p");
  const rp = o.linear(o.relative(L, F), "edge_feature_net.linear_relpos");
  let z = o.edgeFeatures(
    sp,
    rp,
    trans,
    sc,
    diffuse,
    L,
    F,
    B,
    cfg.edge_features.embed_diffuse_mask,
  );
  z = o.linear(z, "edge_feature_net.edge_embedder.0", true);
  z = o.linear(z, "edge_feature_net.edge_embedder.2", true);
  z = o.norm(
    o.linear(z, "edge_feature_net.edge_embedder.4"),
    "edge_feature_net.edge_embedder.5",
  );
  z = o.mask(z, mask, L, true);
  s = o.mask(s, mask, L, false);
  let frames = o.frames(rot, trans);
  const keep = [mask, diffuse];
  for (let b = 0; b < a.num_blocks; b++) {
    o.collect([...keep, s, z, frames]);
    const p = `trunk.ipa_${b}`,
      H = a.no_heads,
      K = a.c_hidden,
      Q = a.no_qk_points,
      V = a.no_v_points;
    const q = o.linear(s, p + ".linear_q"),
      kv = o.linear(s, p + ".linear_kv");
    const qp = o.points(o.linear(s, p + ".linear_q_points"), frames, H * Q);
    const kvp = o.points(
      o.linear(s, p + ".linear_kv_points"),
      frames,
      H * (Q + V),
    );
    const bias = o.linear(z, p + ".linear_b");
    const att = o.softmax(
      o.ipaLogits(
        q,
        kv,
        qp,
        kvp,
        bias,
        o.weight(p + ".head_weights"),
        mask,
        L,
        H,
        K,
        Q,
        V,
      ),
      L,
    );
    const dz = o.linear(z, p + ".down_z");
    let v = o.ipaOutput(att, kv, kvp, dz, frames, L, H, K, Q, V, E / 4);
    v = o.mask(o.linear(v, p + ".linear_out"), mask, L, false);
    s = o.norm(o.add(s, v), `trunk.ipa_ln_${b}`);
    const residual = s;
    for (let n = 0; n < a.seq_tfmr_num_layers; n++) {
      const t = `trunk.seq_tfmr_${b}.layers.${n}`;
      const qkv = o.linear(s, t + ".self_attn.in_proj");
      const attn = o.softmax(
        o.attentionLogits(qkv, mask, L, C, a.seq_tfmr_num_heads),
        L,
      );
      const out = o.attentionOutput(attn, qkv, L, C, a.seq_tfmr_num_heads);
      s = o.norm(
        o.add(s, o.linear(out, t + ".self_attn.out_proj")),
        t + ".norm1",
      );
      s = o.norm(
        o.add(s, o.linear(o.linear(s, t + ".linear1", true), t + ".linear2")),
        t + ".norm2",
      );
    }
    s = o.add(residual, o.linear(s, `trunk.post_tfmr_${b}`));
    const t = `trunk.node_transition_${b}`;
    s = o.norm(
      o.add(
        s,
        o.linear(
          o.linear(o.linear(s, t + ".linear_1", true), t + ".linear_2", true),
          t + ".linear_3",
        ),
      ),
      t + ".ln",
    );
    s = o.mask(s, mask, L, false);
    frames = o.updateFrames(
      frames,
      o.linear(s, `trunk.bb_update_${b}.linear`),
      mask,
      diffuse,
    );
    if (b < a.num_blocks - 1) {
      const e = `trunk.edge_transition_${b}`;
      const ep = o.pairConcat(z, o.linear(s, e + ".initial_embed"), L);
      z = o.norm(
        o.linear(
          o.add(
            ep,
            o.linear(o.linear(ep, e + ".trunk.0", true), e + ".trunk.2", true),
          ),
          e + ".final_layer",
        ),
        e + ".layer_norm",
      );
      z = o.mask(z, mask, L, true);
    }
    progress?.({ phase: "block", block: b + 1, total: a.num_blocks });
  }
  const pred_trans = o.frameOutput(frames, false),
    pred_rotmats = o.frameOutput(frames, true);
  o.collect([pred_trans, pred_rotmats]);
  return { pred_trans, pred_rotmats };
}

export function normalizeInput(x) {
  const L = x.length ?? x.trans_t?.length / 3;
  if (!Number.isSafeInteger(L) || L < 1)
    throw new RangeError("length must be a positive integer");
  const out = {
    ...x,
    length: L,
    so3_t: x.so3_t ?? x.t ?? 0.01,
    r3_t: x.r3_t ?? x.t ?? 0.01,
  };
  for (const [k, n, init] of [
    ["trans_t", 3],
    ["rotmats_t", 9],
    ["res_mask", 1, () => 1],
    ["diffuse_mask", 1, () => 1],
    ["res_idx", 1, (i) => i],
  ]) {
    const data =
      x[k] ??
      (init ? Float32Array.from({ length: L }, (_, i) => init(i)) : null);
    if (
      !data ||
      data.length !== L * n ||
      Array.from(data).some((v) => !Number.isFinite(v))
    )
      throw new RangeError(`Invalid ${k}`);
    out[k] = Float32Array.from(data);
  }
  if (
    out.res_mask.some((v) => v !== 0 && v !== 1) ||
    out.diffuse_mask.some((v) => v !== 0 && v !== 1) ||
    !out.res_mask.some((v) => v)
  )
    throw new RangeError(
      "Masks must be binary with at least one present residue",
    );
  if (
    out.trans_sc &&
    (out.trans_sc.length !== 3 * L ||
      Array.from(out.trans_sc).some((v) => !Number.isFinite(v)))
  )
    throw new RangeError("Invalid trans_sc");
  for (const t of [out.so3_t, out.r3_t])
    if (!Number.isFinite(t) || t < 0 || t > 1)
      throw new RangeError("Time must lie in [0,1]");
  return out;
}
