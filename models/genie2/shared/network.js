// Handwritten modular Denoiser, translated from Genie 2 commit 9a954578.
// Copyright 2021 AlQuraishi Laboratory; Copyright 2021 DeepMind Technologies Limited.
// Modified for independent JavaScript/WGSL inference; Apache-2.0.
import { portable } from "./portable.js";
import { orientations } from "./quaternion.js";
import { applyPoints, invertPoints, backboneUpdate } from "./geometry.js";
export async function denoise(
  o,
  w,
  c,
  frames,
  f,
  timestep,
  totalSteps,
  { signal, onStage } = {},
) {
  const L = frames.trans.shape[0],
    C = c.c_s,
    P = c.c_p;
  const lin = (x, n) => o.linear(x, w[n + ".weight"], w[n + ".bias"]);
  const norm = (x, n) => o.norm(x, w[n + ".weight"], w[n + ".bias"]);
  const notify = async (n, x) => {
    signal?.throwIfAborted();
    if (onStage) await onStage(n, x);
  };
  const keep = [...Object.values(f), frames.trans, frames.rots];
  const collect = async (...xs) => {
    signal?.throwIfAborted();
    await o.checkpoint([...keep, ...xs]);
  };
  const pm = o.einsum("i,j->ij", f.residue_mask, f.residue_mask),
    pm3 = o.reshape(pm, [L, L, 1]);
  keep.push(pm);
  let rots = frames.rots,
    trans = o.mul(frames.trans, c.rescale);
  // Sinusoidal features use the upstream (non-transformer) frequency convention.
  const width = c.c_pos_emb + c.c_chain_emb + c.c_timestep_emb + 23;
  const emb = portable(
    o,
    [L, width],
    [
      f.residue_index,
      f.chain_index,
      f.aatype,
      f.fixed_sequence_mask,
      f.interface_mask,
      o.tensor([2], [timestep, totalSteps]),
    ],
    `
 let r=u32(floor(f32(i)/${width}.0));let j=i%${width}u;var k=j;var D=${c.c_pos_emb}.0;var N=${c.max_n_res}.0;var v=a0[r];
 if(j>=${c.c_pos_emb}u&&j<${c.c_pos_emb + c.c_chain_emb}u){k=j-${c.c_pos_emb}u;D=${c.c_chain_emb}.0;N=${c.max_n_chain}.0;v=a1[r];}
 if(j>=${c.c_pos_emb + c.c_chain_emb}u){k=j-${c.c_pos_emb + c.c_chain_emb}u;D=${c.c_timestep_emb}.0;N=a5[1u];v=a5[0u];}
 if(j<${width - 23}u){var exponent=2.0*f32(k)/D;if(k%2u==1u){exponent=2.0*f32(k+1u)/D;}let arg=(v*3.141592653589793)/pow(N,exponent);if(k%2u==0u){o[i]=cos(arg);}else{o[i]=sin(arg);}}
 else if(j<${width - 3}u){o[i]=a2[r*20u+j-${width - 23}u]*a3[r];}else if(j<${width - 1}u){o[i]=a3[r];}else{o[i]=a4[r];}`,
  );
  let s = o.mul(
    lin(emb, "single_feature_net.linear"),
    o.reshape(f.residue_mask, [L, 1]),
  );
  await notify("single", s);
  const pi = lin(s, "pair_feature_net.linear_s_p_i"),
    pj = lin(s, "pair_feature_net.linear_s_p_j");
  let p = o.add(o.reshape(pi, [L, 1, P]), o.reshape(pj, [1, L, P]));
  const R = 2 * c.relpos_k + 3;
  const rel = portable(
    o,
    [L, L, R],
    [f.residue_index, f.chain_index],
    `
 let pair=u32(floor(f32(i)/${R}.0));let a=u32(floor(f32(pair)/${L}.0));let b=pair%${L}u;let k=i%${R}u;var same=0.0;var bin=${R - 2}.0;
 if(a1[a]==a1[b]){same=1.0;bin=min(max(a0[a]-a0[b]+${c.relpos_k}.0,0.0),${2 * c.relpos_k}.0);}
 o[i]=0.0;if(k==${R - 1}u){o[i]=same;}else if(k==u32(bin)){o[i]=1.0;}`,
  );
  p = o.add(p, lin(rel, "pair_feature_net.linear_relpos"));
  const distanceEncoding = (x, mask) =>
    portable(
      o,
      [L, L, c.template_dist_n_bin],
      [x, mask],
      `
 let pair=u32(floor(f32(i)/${c.template_dist_n_bin}.0));let a=u32(floor(f32(pair)/${L}.0));let b=pair%${L}u;let k=i%${c.template_dist_n_bin}u;var dd=1e-10;
 for(var j=0u;j<3u;j++){let z=a0[a*3u+j]-a0[b*3u+j];dd=dd+z*z;}
 let distance=sqrt(dd);var best=0u;var delta=3.402823e38;
 for(var j=0u;j<${c.template_dist_n_bin}u;j++){let d=abs(distance-(${c.template_dist_min}.0+f32(j)*${c.template_dist_step}));if(d<delta){delta=d;best=j;}}
 o[i]=0.0;if(k==best){o[i]=a1[a]*a1[b];}`,
    );
  const q = orientations(o, rots, f.residue_mask);
  await notify("orientations", q);
  const fixed = o.reshape(f.fixed_structure_mask, [L, L, 1]);
  const templ = o.cat([
    distanceEncoding(trans, f.residue_mask),
    q,
    fixed,
    fixed,
  ]);
  p = o.add(p, lin(templ, "pair_feature_net.linear_template"));
  const motif = o.cat([
    o.mul(distanceEncoding(f.atom_positions, f.fixed_sequence_mask), fixed),
    fixed,
    fixed,
  ]);
  p = o.mul(
    o.add(p, lin(motif, "pair_feature_net.linear_motif_template")),
    pm3,
  );
  await notify("pair", p);
  await collect(s, p, rots, trans);
  for (let layer = 0; layer < c.n_pair_transform_layer; layer++) {
    const prefix = `pair_transform_net.net.${layer}`;
    for (const outgoing of [true, false]) {
      const n = prefix + (outgoing ? ".tri_mul_out" : ".tri_mul_in"),
        z = norm(p, n + ".layer_norm_in");
      const a = o.mul(
        o.mul(lin(z, n + ".linear_a_p"), o.sigmoid(lin(z, n + ".linear_a_g"))),
        pm3,
      );
      const b = o.mul(
        o.mul(lin(z, n + ".linear_b_p"), o.sigmoid(lin(z, n + ".linear_b_g"))),
        pm3,
      );
      const product = o.einsum(
        outgoing ? "ikc,jkc->ijc" : "kic,kjc->ijc",
        a,
        b,
      );
      const x = lin(norm(product, n + ".layer_norm_out"), n + ".linear_z");
      p = o.add(p, o.mul(x, o.sigmoid(lin(z, n + ".linear_g"))));
      await collect(s, p, rots, trans);
    }
    const n = prefix + ".pair_transition";
    const z = lin(
      o.relu(lin(norm(p, n + ".layer_norm"), n + ".linear_1")),
      n + ".linear_2",
    );
    p = o.mul(o.add(p, o.mul(z, pm3)), pm3);
    await notify("pair" + layer, p);
    await collect(s, p, rots, trans);
  }
  await notify("pair_final", p);
  const H = c.n_head_ipa,
    K = c.c_hidden_ipa,
    Q = c.n_qk_point,
    V = c.n_v_point;
  for (let block = 0; block < c.n_structure_block; block++)
    for (let layer = 0; layer < c.n_structure_layer; layer++) {
      const n = `structure_net.net.${layer}`,
        ip = n + ".ipa";
      const query = o.reshape(lin(s, ip + ".linear_q"), [L, H, K]),
        kv = o.reshape(lin(s, ip + ".linear_kv"), [L, H, 2 * K]);
      const key = o.slice(kv, 2, 0, K),
        value = o.slice(kv, 2, K, 2 * K);
      const qp = applyPoints(
        o,
        lin(s, ip + ".linear_q_points"),
        rots,
        trans,
        H,
        Q,
      );
      const kvp = applyPoints(
        o,
        lin(s, ip + ".linear_kv_points"),
        rots,
        trans,
        H,
        Q + V,
      );
      const kp = o.slice(kvp, 2, 0, Q),
        vp = o.slice(kvp, 2, Q, Q + V);
      const bias = lin(p, ip + ".linear_b");
      let logits = o.mul(
        o.einsum("ihc,jhc->hij", query, key),
        Math.sqrt(1 / (3 * K)),
      );
      logits = o.add(
        logits,
        o.mul(o.permute(bias, [2, 0, 1]), Math.sqrt(1 / 3)),
      );
      const pointLogits = portable(
        o,
        [H, L, L],
        [qp, kp, w[ip + ".head_weights"]],
        `
   let h=u32(floor(f32(i)/${L * L}.0));let pair=i%${L * L}u;let a=u32(floor(f32(pair)/${L}.0));let b=pair%${L}u;
   let hw=log(1.0+exp(a2[h]))*${Math.sqrt(1 / (3 * ((Q * 9) / 2)))};var sum=0.0;
   for(var pt=0u;pt<${Q}u;pt++){var dd=0.0;for(var j=0u;j<3u;j++){let d=a0[((a*${H}u+h)*${Q}u+pt)*3u+j]-a1[((b*${H}u+h)*${Q}u+pt)*3u+j];dd=dd+d*d;}sum=sum+dd*hw;}
   o[i]=-0.5*sum;`,
      );
      logits = o.add(logits, pointLogits);
      logits = o.add(logits, o.mul(o.sub(o.reshape(pm, [1, L, L]), 1), 1e5));
      const attention = o.softmax(logits);
      const scalar = o.reshape(o.einsum("hij,jhc->ihc", attention, value), [
        L,
        H * K,
      ]);
      const outPoints = invertPoints(
        o,
        o.einsum("hij,jhpc->ihpc", attention, vp),
        rots,
        trans,
      );
      const norms = portable(
        o,
        [L, H * V],
        [outPoints],
        `let a=a0[i*3u];let b=a0[i*3u+1u];let c=a0[i*3u+2u];o[i]=sqrt(((a*a+b*b)+c*c)+1e-8);`,
      );
      const pc = [0, 1, 2].map((j) =>
        o.reshape(o.slice(outPoints, 3, j, j + 1), [L, H * V]),
      );
      const pair = o.reshape(o.einsum("hij,ijc->ihc", attention, p), [
        L,
        H * P,
      ]);
      const update = lin(
        o.cat([scalar, ...pc, norms, pair]),
        ip + ".linear_out",
      );
      s = norm(o.add(s, update), n + ".ipa_layer_norm");
      for (let j = 0; j < c.n_structure_transition_layer; j++) {
        const a = n + `.transition.layers.${j}`;
        s = o.add(
          s,
          lin(
            o.relu(lin(o.relu(lin(s, a + ".linear_1")), a + ".linear_2")),
            a + ".linear_3",
          ),
        );
      }
      s = norm(s, n + ".transition.layer_norm");
      ({ rots, trans } = backboneUpdate(
        o,
        lin(s, n + ".bb_update.linear"),
        rots,
        trans,
      ));
      await notify(`structure${layer}.single`, s);
      await notify(`structure${layer}.rots`, rots);
      await notify(`structure${layer}.trans`, trans);
      await collect(s, p, rots, trans);
    }
  trans = o.mul(trans, 1 / c.rescale);
  const z = o.sub(frames.trans, trans);
  await notify("z", z);
  return { z, rots, trans, single: s, pair: p };
}
