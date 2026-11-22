// OA_DM_38M uses the sequence-models MSA alphabet, including a separate gap and pad.
export const alphabet = "ACDEFGHIKLMNPQRSTVWYBZXJOU-*#@!";
export const GAP = 26,
  STOP = 27,
  MASK = 28,
  START = 29,
  PAD = 30;

export function normalizeTokens(input) {
  if (typeof input === "string") return encode(input);
  if (!input || !Number.isSafeInteger(input.length))
    throw new TypeError("Expected sequence or token IDs");
  for (const id of input)
    if (!Number.isInteger(id) || id < 0 || id >= 31)
      throw new RangeError("Invalid token ID");
  return Int32Array.from(input);
}
export function decode(input) {
  return Array.from(normalizeTokens(input), (i) => alphabet[i]).join("");
}

/** Encode the contents of one upstream Tokenizer.tokenize([sequence]) row. */
export function encode(sequence) {
  if (typeof sequence !== "string")
    throw new TypeError("Expected one sequence string");
  const ids = new Int32Array(sequence.length);
  for (let position = 0; position < sequence.length; position++) {
    const id = alphabet.indexOf(sequence[position]);
    if (id < 0) {
      throw new RangeError(
        `Unknown token at position ${position}: ${JSON.stringify(sequence[position])}`,
      );
    }
    ids[position] = id;
  }
  return ids;
}
