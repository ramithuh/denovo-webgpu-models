// Dynamic, hand-written architecture. Operators have interchangeable CPU/WGSL
// implementations. No traced graph, fixed length, or fixed sampling step count.
export function network(o, input, { trace } = {}) {
  const { length: L } = input,
    N = L + 10;
  const mark = (name, t) => {
    trace?.(name, t);
    return t;
  };
  const scoped = (fn) => o.scope(fn);
  const linear = (x, n) => o.linear(x, n);
  function transition(x, p) {
    return scoped(() =>
      linear(o.swiglu(linear(x, p + ".swish_linear.0")), p + ".linear_out"),
    );
  }
  function adaln(x, c, p) {
    return scoped(() => {
      const cn = o.norm(c, p + ".norm_cond"),
        gamma = linear(cn, p + ".to_gamma.0"),
        beta = linear(cn, p + ".to_beta");
      return o.adapt(o.norm(x), gamma, beta);
    });
  }
  function scaledResidual(x, delta, c, p) {
    return scoped(() =>
      o.scaledAdd(x, delta, linear(c, p + ".to_adaln_zero_gamma.0")),
    );
  }
  const time = o.time(input.t),
    fold = o.fold(input.fold),
    features = o.features(input);
  let c = mark(
    "cond_factory",
    linear(o.concat(time, fold), "cond_factory.linear_out"),
  );
  c = mark("transition_c_1", transition(c, "transition_c_1"));
  c = mark("transition_c_2", transition(c, "transition_c_2"));
  let x = mark(
    "init_repr_factory",
    linear(features, "init_repr_factory.linear_out"),
  );
  x = o.add(x, linear(o.coordinates(input), "linear_3d_embed"));
  let pair = scoped(() => {
    const p = "pair_repr_builder";
    const initial = o.norm(
      linear(o.pairFeatures(input), p + ".init_repr_factory.linear_out"),
      p + ".init_repr_factory.ln_out",
    );
    const pc = o.norm(
      linear(time, p + ".cond_factory.linear_out"),
      p + ".cond_factory.ln_out",
    );
    return adaln(initial, pc, p + ".adaln");
  });
  pair = mark("pair_repr_builder", pair);
  // Zero conditioning for registers; one repeated conditioning row for residues.
  c = o.extendCond(c, L);
  x = o.extendSequence(x, L);
  pair = o.extendPair(pair, L);
  o.retain([c, x, pair]);
  for (let i = 0; i < 15; i++) {
    const p = `transformer_layers.${i}`;
    x = scoped(() => {
      const a = p + ".mhba",
        m = a + ".mha";
      const node = o.norm(adaln(x, c, a + ".adaln"), m + ".node_norm");
      const qkv = linear(node, m + ".to_qkv");
      const q = o.norm(o.slice(qkv, 0, 768), m + ".q_layer_norm");
      const k = o.norm(o.slice(qkv, 768, 768), m + ".k_layer_norm");
      const v = o.slice(qkv, 1536, 768),
        g = linear(node, m + ".to_g");
      const bias = linear(o.norm(pair, m + ".pair_norm"), m + ".to_bias");
      const out = linear(
        o.gatedAttention(q, k, v, g, bias, N),
        m + ".to_out_node",
      );
      return scaledResidual(x, out, c, a + ".scale_output");
    });
    o.retain([c, x, pair]);
    x = scoped(() => {
      const pfx = p + ".transition";
      return scaledResidual(
        x,
        transition(adaln(x, c, pfx + ".adaln"), pfx + ".transition"),
        c,
        pfx + ".scale_output",
      );
    });
    mark(p, x);
    o.retain([c, x, pair]);
  }
  return scoped(() =>
    linear(
      o.norm(o.removeRegisters(x, L), "coors_3d_decoder.0"),
      "coors_3d_decoder.1",
    ),
  );
}
export function validateInput(
  coordinates,
  { t = 0.5, xSC, fold, residueIndices, chainBreaks, signal } = {},
) {
  signal?.throwIfAborted();
  const x =
      coordinates instanceof Float32Array
        ? coordinates
        : Float32Array.from(coordinates ?? []),
    length = x.length / 3;
  if (
    !Number.isSafeInteger(length) ||
    length < 1 ||
    x.some((v) => !Number.isFinite(v))
  )
    throw RangeError("Expected finite XYZ coordinates");
  if (!Number.isFinite(t) || t < 0 || Math.fround(t) >= 1)
    throw RangeError("t must be in [0,1)");
  const check = (a, n, label) => {
    if (
      a != null &&
      (a.length !== n ||
        Array.from(a).some((v) => !Number.isFinite(Math.fround(v))))
    )
      throw RangeError(`Invalid ${label}`);
  };
  check(xSC, x.length, "self conditioning");
  check(residueIndices, length, "residue indices");
  check(chainBreaks, length, "chain breaks");
  fold ??= [5, 43, 1472];
  check(fold, 3, "fold indices");
  if (
    fold.some((v, i) => !Number.isInteger(v) || v < 0 || v >= [6, 44, 1473][i])
  )
    throw RangeError("Fold index out of range");
  return {
    x,
    length,
    t: Math.fround(t),
    xSC,
    fold,
    residueIndices,
    chainBreaks,
  };
}
