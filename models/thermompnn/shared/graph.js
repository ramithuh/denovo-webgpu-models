// Backend-neutral scheduling. The WebGPU backend implements every operation in WGSL.
export async function executeGraph(o, input) {
  const P = "prot_mpnn.",
    L = input.length,
    K = Math.min(48, L),
    R = L * K;
  const { geometry, indices } = o.geometry(input);
  o.trace("geometry", geometry);
  o.trace("E_idx", indices);
  let features = o.dense(geometry, P + "features.edge_embedding");
  o.free(geometry);
  let E = o.norm(features, null, P + "features.norm_edges");
  o.free(features);
  o.trace("features", E);
  let edges = o.dense(E, P + "W_e");
  o.free(E);
  o.trace("edge_projection", edges);
  let nodes = o.zeros(L, 128);
  for (let i = 0; i < 3; i++) {
    const p = P + `encoder_layers.${i}.`;
    let msg = o.concat(nodes, edges, indices, null, input, false);
    let a = o.dense(msg, p + "W1", "gelu");
    o.free(msg);
    let b = o.dense(a, p + "W2", "gelu");
    o.free(a);
    msg = o.dense(b, p + "W3");
    o.free(b);
    let delta = o.reduce(msg, indices, input, true);
    o.free(msg);
    let v = o.norm(nodes, delta, p + "norm1");
    o.free(nodes);
    o.free(delta);
    a = o.dense(v, p + "dense.W_in", "gelu");
    b = o.dense(a, p + "dense.W_out");
    o.free(a);
    nodes = o.norm(v, b, p + "norm2", input);
    o.free(v);
    o.free(b);
    msg = o.concat(nodes, edges, indices, null, input, false);
    a = o.dense(msg, p + "W11", "gelu");
    o.free(msg);
    b = o.dense(a, p + "W12", "gelu");
    o.free(a);
    msg = o.dense(b, p + "W13");
    o.free(b);
    E = o.norm(edges, msg, p + "norm3");
    o.free(edges);
    o.free(msg);
    edges = E;
    o.trace(`encoder_${i}`, nodes);
    o.trace(`encoder_edges_${i}`, edges);
    await o.checkpoint(`encoder_${i}`);
  }
  const embedding = o.embed(input, P + "W_s.weight");
  o.trace("sequence_embedding", embedding);
  let penultimate;
  for (let i = 0; i < 3; i++) {
    const p = P + `decoder_layers.${i}.`;
    let msg = o.concat(nodes, edges, indices, embedding, input, true);
    let a = o.dense(msg, p + "W1", "gelu");
    o.free(msg);
    let b = o.dense(a, p + "W2", "gelu");
    o.free(a);
    msg = o.dense(b, p + "W3");
    o.free(b);
    const delta = o.reduce(msg, indices, input, false);
    o.free(msg);
    const v = o.norm(nodes, delta, p + "norm1");
    if (i !== 2) o.free(nodes);
    o.free(delta);
    a = o.dense(v, p + "dense.W_in", "gelu");
    b = o.dense(a, p + "dense.W_out");
    o.free(a);
    nodes = o.norm(v, b, p + "norm2", input);
    o.free(v);
    o.free(b);
    o.trace(`decoder_${i}`, nodes);
    if (i === 1) penultimate = nodes;
    await o.checkpoint(`decoder_${i}`);
  }
  o.free(edges);
  o.free(indices);
  const logits = o.dense(nodes, P + "W_out");
  o.trace("logits", logits);
  o.free(logits);
  let x = o.join([nodes, penultimate, embedding]);
  o.free(nodes);
  o.free(penultimate);
  o.free(embedding);
  let a = o.dense(x, "light_attention.feature_convolution");
  o.free(x);
  o.trace("light_attention", a);
  for (let i = 0; i < 3; i++) {
    x = o.dense(a, `both_out.${1 + 2 * i}`, "none", true);
    o.free(a);
    a = x;
    o.trace(`head_${i}`, a);
  }
  const scores = o.calibrate(a);
  o.free(a);
  o.trace("scores", scores);
  const ddg = o.subtract(scores, input);
  o.free(scores);
  o.trace("ddg", ddg);
  return ddg;
}
