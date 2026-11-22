/** ESM character vocabulary, with explicit BOS/EOS and upstream mask shorthand. */
export const VOCAB = [
  "<cls>",
  "<pad>",
  "<eos>",
  "<unk>",
  ..."LAGVSERTIDPKQNFYMHWCXBUZO",
  ".",
  "-",
  "|",
  "<mask>",
];
export const ALPHABET = "ACDEFGHIKLMNPQRSTVWY";
export const AA_IDS = Int32Array.from(ALPHABET, (c) => VOCAB.indexOf(c));
export function tokenize(sequence, maxLength = 1022) {
  if (typeof sequence !== "string")
    throw new TypeError("Expected sequence text");
  const pieces =
    sequence
      .replaceAll("_", "<mask>")
      .match(/<cls>|<pad>|<eos>|<unk>|<mask>|[\s\S]/g) ?? [];
  if (!pieces.length || pieces.length > maxLength)
    throw new RangeError(`Expected 1–${maxLength} sequence tokens`);
  return Int32Array.from([
    0,
    ...pieces.map((c) => (VOCAB.indexOf(c) < 0 ? 3 : VOCAB.indexOf(c))),
    2,
  ]);
}
export function prepareInput(input, maxLength = 1022, sequenceIds) {
  let tokens, residues;
  if (typeof input === "string") {
    if (!/^(?:[A-Z._|\-]|<mask>|<unk>)+$/.test(input))
      throw new TypeError(
        "Use uppercase sequence letters or mask tokens; structure files are not sequence inputs",
      );
    tokens = tokenize(input, maxLength);
    residues = Array.from(tokens.subarray(1, -1), (i) =>
      i === 32 ? "_" : i < 4 ? "X" : VOCAB[i],
    ).join("");
  } else if (
    input instanceof Int32Array ||
    input instanceof Uint32Array ||
    Array.isArray(input)
  ) {
    if (
      input.length < 3 ||
      input.length > maxLength + 2 ||
      Array.from(input).some((i) => !Number.isInteger(i) || i < 0 || i > 32)
    )
      throw new RangeError("Invalid ESM token IDs");
    tokens = Int32Array.from(input);
    residues = Array.from(tokens.subarray(1, -1), (i) =>
      i === 32 ? "_" : i < 4 ? "X" : VOCAB[i],
    ).join("");
  } else throw new TypeError("Expected sequence text or token array");
  const ids =
    sequenceIds === undefined
      ? Int32Array.from(tokens, (t) => (t !== 1 ? 1 : 0))
      : Int32Array.from(sequenceIds);
  if (
    ids.length !== tokens.length ||
    (sequenceIds !== undefined &&
      Array.from(sequenceIds).some(
        (i) => !Number.isSafeInteger(i) || i < -2147483648 || i > 2147483647,
      ))
  )
    throw new RangeError("sequenceIds must contain one int32 per token");
  return { tokens, sequenceIds: ids, residues };
}
export function canonical(sequence) {
  if (
    typeof sequence !== "string" ||
    !sequence.length ||
    !/^[ACDEFGHIKLMNPQRSTVWY]+$/.test(sequence)
  )
    throw new TypeError("Scoring requires a canonical wild-type sequence");
  return sequence;
}
export function validatePositions(positions, length) {
  if (
    !Array.isArray(positions) ||
    !positions.length ||
    new Set(positions).size !== positions.length ||
    positions.some((p) => !Number.isInteger(p) || p < 0 || p >= length)
  )
    throw new RangeError("positions must be unique zero-based residue indices");
  return positions;
}
export function scoreMutations(sequence, llr, mutations) {
  canonical(sequence);
  if (
    llr.length !== sequence.length * 20 ||
    !Array.isArray(mutations) ||
    !mutations.length
  )
    throw new RangeError("Invalid LLR matrix or mutations");
  validatePositions(
    mutations.map((m) => m.position),
    sequence.length,
  );
  return mutations.reduce((score, { position, from, to }) => {
    if (
      sequence[position] !== from ||
      !ALPHABET.includes(to) ||
      to.length !== 1
    )
      throw new RangeError("Mutation does not match the wild type");
    return score + llr[position * 20 + ALPHABET.indexOf(to)];
  }, 0);
}
