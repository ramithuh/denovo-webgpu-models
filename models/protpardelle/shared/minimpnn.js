export function miniMPNN(ops, config, input, capture = () => {}) {
  const p = "mpnn_model.",
    mp = p + "mpnn_net.mpnn.",
    n = input.seqMask.length,
    d = config.model.mpnn_model.n_channel;
  const noise = ops.noise(
    { seqMask: [1], noiseLevel: input.noiseLevel.slice(0, 1) },
    d,
  );
  const t = ops.linear(
      ops.silu(ops.linear(noise, p + "noise_block.block.1")),
      p + "noise_block.block.3",
    ),
    time = ops.silu(t);
  capture("mpnn.noise", t);
  const { features, indices } = ops.graphFeatures(input, mp + "features.");
  capture("mpnn.features.raw", features);
  capture("mpnn.neighbors", indices);
  let edge = ops.norm(
    ops.linear(features, mp + "features.edge_embedding"),
    mp + "features.norm_edges",
  );
  capture("mpnn.features", edge);
  edge = ops.linear(edge, mp + "W_e");
  let node = ops.repeatRows(ops.linear(time, mp + "time_block.1"), n);
  for (let i = 0; i < config.model.mpnn_model.n_layers; i++) {
    const prefix = mp + `encoder_layers.${i}.`;
    let message = ops.gelu(
      ops.linear(
        ops.gelu(
          ops.linear(ops.neighbors(node, edge, indices, true), prefix + "W1"),
        ),
        prefix + "W2",
      ),
    );
    message = ops.modulate(
      message,
      ops.linear(time, prefix + "time_block1.2"),
      "affine",
    );
    const dh = ops.sumNeighbors(
      ops.linear(message, prefix + "W3"),
      indices,
      input.seqMask,
      true,
    );
    node = ops.norm(ops.add(node, dh), prefix + "norm1");
    node = ops.mask(
      ops.norm(
        ops.add(
          node,
          ops.linear(
            ops.gelu(ops.linear(node, prefix + "dense.W_in")),
            prefix + "dense.W_out",
          ),
        ),
        prefix + "norm2",
      ),
      input.seqMask,
    );
    message = ops.gelu(
      ops.linear(
        ops.gelu(
          ops.linear(ops.neighbors(node, edge, indices, true), prefix + "W11"),
        ),
        prefix + "W12",
      ),
    );
    message = ops.modulate(
      message,
      ops.linear(time, prefix + "time_block2.2"),
      "affine",
    );
    edge = ops.norm(
      ops.add(edge, ops.linear(message, prefix + "W13")),
      prefix + "norm3",
    );
    capture(`mpnn.encoder.${i}.node`, node);
    capture(`mpnn.encoder.${i}.edge`, edge);
  }
  let sequence = ops.zeros(n, d);
  if (input.seqSelfCond && config.model.mpnn_model.use_self_conditioning)
    sequence = ops.linear(
      ops.exp(ops.fromArray(input.seqSelfCond, n, 21)),
      p + "token_embedding",
    );
  if (input.seqCropCond)
    sequence = ops.add(
      sequence,
      ops.linear(
        ops.fromArray(input.seqCropCond, n, 21),
        p + "token_embedding",
      ),
    );
  const edgeSequence = ops.neighbors(sequence, edge, indices, false);
  for (let i = 0; i < config.model.mpnn_model.n_layers; i++) {
    const prefix = mp + `decoder_layers.${i}.`;
    // Noncausal MiniMPNN: only query masks are applied before the decoder.
    let ev = ops.neighbors(node, edgeSequence, indices, false);
    ev = ops.maskEdges(ev, indices, input.seqMask);
    let message = ops.gelu(
      ops.linear(
        ops.gelu(
          ops.linear(ops.centerConcat(node, ev, indices), prefix + "W1"),
        ),
        prefix + "W2",
      ),
    );
    message = ops.modulate(
      message,
      ops.linear(time, prefix + "time_block.2"),
      "affine",
    );
    node = ops.norm(
      ops.add(
        node,
        ops.sumNeighbors(
          ops.linear(message, prefix + "W3"),
          indices,
          input.seqMask,
          false,
        ),
      ),
      prefix + "norm1",
    );
    node = ops.mask(
      ops.norm(
        ops.add(
          node,
          ops.linear(
            ops.gelu(ops.linear(node, prefix + "dense.W_in")),
            prefix + "dense.W_out",
          ),
        ),
        prefix + "norm2",
      ),
      input.seqMask,
    );
    capture(`mpnn.decoder.${i}`, node);
  }
  const logits = ops.linear(node, p + "proj_out");
  capture("mpnn.logits", logits);
  const probabilities = ops.logSoftmax(logits);
  capture("mpnn.logprobs", probabilities);
  return probabilities;
}
