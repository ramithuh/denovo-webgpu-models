import {
  canonical,
  tokenize,
  validatePositions,
  ALPHABET,
  VOCAB,
} from "./input.js";
/** Task orchestration only. Neural and probability arithmetic stays in each backend. */
export async function runTask(
  engine,
  sequence,
  {
    mode = "embeddings",
    positions,
    signal,
    onProgress,
    onTrace,
    ...options
  } = {},
) {
  if (
    ![
      "embeddings",
      "logits",
      "wildtype",
      "masked",
      "naturalness",
      "fill",
    ].includes(mode)
  )
    throw new RangeError("Unknown inference mode");
  const start = performance.now();
  signal?.throwIfAborted();
  if (["masked", "naturalness", "wildtype", "fill"].includes(mode))
    canonical(sequence);
  if (mode === "fill") {
    validatePositions(positions, sequence.length);
    const tokens = tokenize(sequence, engine.config.max_sequence_length),
      wildTypeTokens = tokens.slice();
    for (const p of positions) tokens[p + 1] = 32;
    const out = await engine.forward(tokens, {
      ...options,
      signal,
      wildTypeTokens,
    });
    const filled = sequence.split("");
    for (const p of positions) filled[p] = VOCAB[out.predictions[p]];
    return {
      ...out,
      mode,
      positions,
      residues: sequence,
      maskedSequence: out.residues,
      filledSequence: filled.join(""),
      timingMs: performance.now() - start,
    };
  }
  if (mode !== "masked" && mode !== "naturalness")
    return {
      ...(await engine.forward(sequence, { ...options, signal })),
      mode,
    };
  const L = sequence.length,
    tokens = tokenize(sequence, engine.config.max_sequence_length);
  const intermediateRows = options.returnIntermediates ? [] : undefined;
  const llr = new Float32Array(L * 20),
    wtLogProbs = new Float32Array(L),
    maskedLogits = new Float32Array(L * 64);
  for (let p = 0; p < L; p++) {
    signal?.throwIfAborted();
    const masked = tokens.slice();
    masked[p + 1] = 32;
    const out = await engine.forward(masked, {
      ...options,
      signal,
      wildTypeTokens: tokens,
      traceRow: p + 1,
      rowOnly: true,
      returnIntermediates: !!onTrace || !!intermediateRows,
    });
    const llrOff = out.llr.length === 20 ? 0 : p * 20;
    llr.set(out.llr.subarray(llrOff, llrOff + 20), p * 20);
    wtLogProbs[p] =
      out.wtLogProbs.length === 1 ? out.wtLogProbs[0] : out.wtLogProbs[p];
    const logOff = out.logits.length === 64 ? 0 : (p + 1) * 64;
    maskedLogits.set(out.logits.subarray(logOff, logOff + 64), p * 64);
    if (intermediateRows) intermediateRows.push(out.intermediates);
    if (onTrace) await onTrace(p, out.intermediates);
    onProgress?.({ phase: "mask-sweep", loaded: p + 1, total: L });
  }
  signal?.throwIfAborted();
  const scores = await engine.summarize(wtLogProbs, { signal });
  return {
    mode,
    residues: sequence,
    alphabet: ALPHABET,
    llr,
    wtLogProbs,
    maskedLogits,
    ...(intermediateRows ? { intermediates: intermediateRows } : {}),
    ...scores,
    timingMs: performance.now() - start,
  };
}
