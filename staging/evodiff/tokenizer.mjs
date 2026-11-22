// OA_DM_38M uses the sequence-models MSA alphabet, including a separate gap and pad.
export const alphabet = 'ACDEFGHIKLMNPQRSTVWYBZXJOU-*#@!';

/** Encode the contents of one upstream Tokenizer.tokenize([sequence]) row. */
export function encode(sequence) {
  if (typeof sequence !== 'string') throw new TypeError('Expected one sequence string');
  const ids = new Int32Array(sequence.length);
  for (let position = 0; position < sequence.length; position++) {
    const id = alphabet.indexOf(sequence[position]);
    if (id < 0) {
      throw new RangeError(`Unknown token at position ${position}: ${JSON.stringify(sequence[position])}`);
    }
    ids[position] = id;
  }
  return ids;
}
