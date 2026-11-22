// The neural graph is shared by independent CPU and WGSL operation backends.
export function denoise(ops, config, input, capture = () => {}) {
  const m = config.model,
    s = m.struct_model,
    d = s.n_channel,
    atoms = s.n_atoms;
  if (s.uvit.patch_size !== 1 || s.uvit.n_filt_per_layer.length)
    throw new Error(
      "Convolutional/patch UViT requires a separate audited graph",
    );
  const prefix = "struct_model.",
    net = prefix + "net.",
    trans = net + "transformer.";
  const n = input.seqMask.length;
  const noise = ops.noise(input, d);
  capture("noise_embedding", noise);
  const t = ops.linear(
    ops.silu(ops.linear(noise, prefix + "noise_block.block.1")),
    prefix + "noise_block.block.3",
  );
  capture("noise_conditioning", t);
  const time = ops.silu(t);
  const embedding = ops.features(input, config);
  capture("input_features", embedding);
  let x = ops.norm(
    ops.linear(embedding, net + "to_patch_embedding.1"),
    net + "to_patch_embedding.2",
  );
  x = ops.mask(x, input.seqMask);
  capture("patch_embedding", x);
  for (let i = 0; i < s.uvit.n_layers; i++) {
    ops.beginBlock?.();
    const a = trans + `layers.${i}.0`,
      f = trans + `layers.${i}.1`;
    const at = ops.linear(time, a + ".time_cond.1");
    const qin = ops.mask(
      ops.modulate(ops.norm(x, a + ".norm"), at, "affine"),
      input.seqMask,
    );
    const q = ops.linear(qin, a + ".to_q"),
      kv = ops.linear(x, a + ".to_kv");
    let attention = ops.attention(q, kv, input, config, trans);
    attention = ops.linear(attention, a + ".to_out");
    if (s.arch === "dit") attention = ops.modulate(attention, at, "gate");
    attention = ops.mask(attention, input.seqMask);
    capture(`block.${i}.attention`, attention);
    x = ops.add(x, attention);
    const ft = ops.linear(time, f + ".time_cond.1");
    let ff = ops.norm(x, f + ".norm");
    if (s.arch !== "dit") ff = ops.silu(ops.linear(ff, f + ".linear_in"));
    ff = ops.modulate(ff, ft, "affine");
    if (s.arch === "dit") ff = ops.silu(ops.linear(ff, f + ".linear_in"));
    ff = ops.linear(ff, f + ".linear_out");
    if (s.arch === "dit") ff = ops.modulate(ff, ft, "gate");
    capture(`block.${i}.ff`, ff);
    x = ops.mask(ops.add(x, ff), input.seqMask);
    capture(`block.${i}.hidden`, x);
    ops.endBlock?.([x]);
  }
  const patch = ops.linear(
    ops.norm(x, net + "from_patch.0"),
    net + "from_patch.1",
  );
  capture("output_patch", patch);
  const result = ops.output(patch, input, config);
  capture("denoised_coords", result);
  return result;
}
export function validateInput(input, config) {
  const n = input.seqMask?.length,
    size = n * config.model.struct_model.n_atoms * 3;
  if (!Number.isInteger(n) || n < 1 || n > 2048)
    throw new Error("Expected 1–2048 residues");
  for (const [key, length] of [
    ["coords", size],
    ["seqMask", n],
    ["residueIndex", n],
    ["noiseLevel", n],
  ]) {
    if (
      !input[key] ||
      input[key].length !== length ||
      Array.from(input[key]).some((x) => !Number.isFinite(x))
    )
      throw new Error(`Invalid ${key}`);
  }
  for (const [key, length] of [
    ["chainIndex", n],
    ["hotspotMask", n],
    ["selfCond", size],
    ["cropCond", size],
  ]) {
    if (
      input[key] &&
      (input[key].length !== length ||
        Array.from(input[key]).some((x) => !Number.isFinite(x)))
    )
      throw new Error(`Invalid ${key}`);
  }
  if (
    Array.from(input.seqMask).some((x) => x !== 0 && x !== 1) ||
    !input.seqMask.some(Boolean)
  )
    throw new Error("Invalid sequence mask");
  if (input.noiseLevel.some((x) => x < 0))
    throw new Error("Negative noise level");
  if (
    config.model.struct_model.uvit.position_embedding_type.includes(
      "relchain",
    ) &&
    !input.chainIndex
  )
    throw new Error("chainIndex is required");
}
