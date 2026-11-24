/** Explicit modular network; L and sampling T are runtime arguments. No graph capture. */
export async function forwardNetwork(b, x, t, cath, trace) {
  const L = x.rows,
    N = L + 10;
  const emit = async (name, value) => {
    if (trace) await trace(name, value);
  };
  const time = await b.time(t);
  let c = await b.condition(time, cath);
  c = await b.linear(c, "cond_factory.linear_out");
  await emit("cond_factory", c);
  for (const k of [1, 2])
    c = await b.linear(
      await b.swiglu(await b.linear(c, `transition_c_${k}.swish_linear.0`)),
      `transition_c_${k}.linear_out`,
    );
  await emit("transition_c_2", c);
  c = await b.padCondition(c);
  const initial = await b.linear(
    await b.position(L),
    "init_repr_factory.linear_out",
  );
  await emit("init_repr_factory", initial);
  let s = await b.add(await b.linear(x, "linear_3d_embed"), initial);
  s = await b.registers(s);
  const pp = "pair_repr_builder";
  let pairLin = b.pairLinearInit
    ? await b.pairLinearInit(x, `${pp}.init_repr_factory.linear_out`)
    : await b.linear(await b.pairFeatures(x), `${pp}.init_repr_factory.linear_out`);
  let pair = await b.norm(pairLin, `${pp}.init_repr_factory.ln_out`);
  let pc = await b.norm(
    await b.linear(time, `${pp}.cond_factory.linear_out`),
    `${pp}.cond_factory.ln_out`,
  );
  pair = await adaptive(b, pair, pc, `${pp}.adaln`, 0);
  await emit("pair_repr_builder", pair);
  pair = await b.padPair(pair, L);
  const pStats = b.pairStats ? b.pairStats(pair) : null;
  b.keep([x, s, c, pair, pStats]);
  for (let layer = 0; layer < 15; layer++) {
    b.check();
    const root = `transformer_layers.${layer}`;
    let a = await adaptive(b, s, c, `${root}.mhba.adaln`, 1);
    if (layer === 0) await emit(`${root}.mhba.adaln`, a);
    const ap = `${root}.mhba.mha`;
    a = await b.norm(a, `${ap}.node_norm`);
    if (layer === 0) await emit(`${ap}.node_norm`, a);
    const qkv = await b.linear(a, `${ap}.to_qkv`);
    if (layer === 0) await emit(`${ap}.to_qkv`, qkv);
    const q = await b.norm(qkv, `${ap}.q_layer_norm`, {
      width: 768,
      stride: 2304,
      offset: 0,
    });
    const k = await b.norm(qkv, `${ap}.k_layer_norm`, {
      width: 768,
      stride: 2304,
      offset: 768,
    });
    const g = await b.linear(a, `${ap}.to_g`);
    if (layer === 0) {
      await emit(`${ap}.q_layer_norm`, q);
      await emit(`${ap}.k_layer_norm`, k);
      await emit(`${ap}.to_g`, g);
    }
    const bias = pStats
      ? await b.pairBias(pair, pStats, `${ap}.pair_norm`, `${ap}.to_bias`)
      : await b.linear(await b.norm(pair, `${ap}.pair_norm`), `${ap}.to_bias`);
    if (layer === 0) await emit(`${ap}.to_bias`, bias);
    let h = await b.attention(q, k, qkv, bias, g, N);
    if (layer === 0) await emit(`${ap}.context`, h);
    h = await b.linear(h, `${ap}.to_out_node`);
    if (layer === 0) await emit(`${ap}.to_out_node`, h);
    const hg = await b.linear(
      c,
      `${root}.mhba.scale_output.to_adaln_zero_gamma.0`,
    );
    s = await b.residualGate(s, h, hg);
    b.keep([x, s, c, pair, pStats]);
    a = await adaptive(b, s, c, `${root}.transition.adaln`, 1);
    if (layer === 0) await emit(`${root}.transition.adaln`, a);
    h = await b.linear(
      await b.swiglu(
        await b.linear(a, `${root}.transition.transition.swish_linear.0`),
      ),
      `${root}.transition.transition.linear_out`,
    );
    const tg = await b.linear(
      c,
      `${root}.transition.scale_output.to_adaln_zero_gamma.0`,
    );
    s = await b.residualGate(s, h, tg);
    if (layer === 0 || layer === 14)
      await emit(`transformer_layers.${layer}`, s);
    b.keep([x, s, c, pair, pStats]);
  }
  const out = await b.linear(
    await b.norm(s, "coors_3d_decoder.0"),
    "coors_3d_decoder.1",
  );
  await emit("coors_3d_decoder", await b.unpad(out, 10));
  return b.velocity(x, out, t);
}
async function adaptive(b, x, c, p, mode) {
  const cn = await b.norm(c, `${p}.norm_cond`);
  const gamma = await b.linear(cn, `${p}.to_gamma.0`),
    beta = await b.linear(cn, `${p}.to_beta`);
  if (b.fastAdaptive) return b.fastAdaptive(x, gamma, beta, mode);
  return b.adaptive(await b.norm(x), gamma, beta, mode);
}
export function assertModelManifest(m) {
  if (
    m.upstream !== "dfef6a29d975451144e8b0449c12cbea28b4ae15" ||
    m.checkpoint_sha256 !==
      "57062dbf5b8a22068b65484b3cee389cd99f1b67bd4e2d280c55366ee22d80e7" ||
    m.num_tensors !== 490 ||
    m.total_params !== 191102976
  )
    throw new Error("Bundle is not the pinned SiD-Protein checkpoint");
  for (const [key, value] of Object.entries({
    tokenDim: 768,
    condDim: 512,
    pairDim: 512,
    layers: 15,
    heads: 12,
    registers: 10,
    target: "v",
  }))
    if (m.config?.[key] !== value)
      throw new Error("SiD-Protein configuration mismatch: " + key);
}
