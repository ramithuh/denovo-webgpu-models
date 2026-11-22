export function expectedShapes() {
  const s = {
    registers: [10, 768],
    "linear_3d_embed.weight": [768, 3],
    "init_repr_factory.linear_out.weight": [768, 132],
    "cond_factory.linear_out.weight": [512, 1024],
  };
  const matrix = (p, o, i, bias = false) => {
    s[p + ".weight"] = [o, i];
    if (bias) s[p + ".bias"] = [o];
  };
  const norm = (p, d) => {
    s[p + ".weight"] = [d];
    s[p + ".bias"] = [d];
  };
  const ada = (p, d) => {
    norm(p + ".norm_cond", 512);
    matrix(p + ".to_gamma.0", d, 512, true);
    matrix(p + ".to_beta", d, 512);
  };
  for (const [c, n] of [
    ["C", 6],
    ["A", 44],
    ["T", 1473],
  ])
    matrix(`cond_factory.feat_creators.1.embedding_${c}`, n, 256);
  for (const n of ["transition_c_1", "transition_c_2"]) {
    matrix(n + ".swish_linear.0", 2048, 512);
    matrix(n + ".linear_out", 512, 1024);
  }
  for (const [n, k] of [
    ["init_repr_factory", 319],
    ["cond_factory", 256],
  ]) {
    norm("pair_repr_builder." + n + ".ln_out", 512);
    matrix("pair_repr_builder." + n + ".linear_out", 512, k);
  }
  ada("pair_repr_builder.adaln", 512);
  for (let i = 0; i < 15; i++) {
    const p = `transformer_layers.${i}`;
    for (const part of ["mhba", "transition"]) {
      ada(p + "." + part + ".adaln", 768);
      matrix(
        p + "." + part + ".scale_output.to_adaln_zero_gamma.0",
        768,
        512,
        true,
      );
    }
    const m = p + ".mhba.mha";
    matrix(m + ".to_qkv", 2304, 768, true);
    matrix(m + ".to_g", 768, 768, true);
    matrix(m + ".to_out_node", 768, 768, true);
    matrix(m + ".to_bias", 12, 512);
    for (const n of ["node_norm", "q_layer_norm", "k_layer_norm"])
      norm(m + "." + n, 768);
    norm(m + ".pair_norm", 512);
    matrix(p + ".transition.transition.swish_linear.0", 6144, 768);
    matrix(p + ".transition.transition.linear_out", 768, 3072);
  }
  norm("coors_3d_decoder.0", 768);
  matrix("coors_3d_decoder.1", 3, 768);
  return s;
}
