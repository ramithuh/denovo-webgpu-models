import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { gatherEmbeddings } from './input-embedding.mjs';
import { alphabet, encode } from './tokenizer.mjs';

// Expected IDs captured from the pinned Python Tokenizer.tokenize([sequence]).
const fixture = JSON.parse(readFileSync(new URL('./tokenizer-fixtures.json', import.meta.url), 'utf8'));

test('one-row token IDs match the upstream fixtures across the entire vocabulary', () => {
  for (const [sequence, expected] of fixture.rows) {
    const result = encode(sequence);
    assert.ok(result instanceof Int32Array);
    assert.deepEqual(Array.from(result), expected, sequence);
  }
  const covered = new Set(fixture.rows.flatMap(([, ids]) => ids));
  assert.equal(covered.size, 31);
  assert.equal(alphabet.length, covered.size);
});

test('rejects the same unsupported characters as the upstream tokenizer', () => {
  for (const invalid of fixture.unknown) {
    assert.throws(() => encode(invalid), RangeError);
  }
});

test('GFP residues 1–8 reproduce the CPU input embedding byte for byte', () => {
  const reference = JSON.parse(readFileSync(new URL('./checkpoint-reference.json', import.meta.url), 'utf8'));
  const { sequence, ids, rows, output_float32_le_sha256 } = reference.input_embedding_fixture;
  const [vocab, width] = reference.embedding_shape;
  const table = new Float32Array(vocab * width).fill(NaN);
  for (const [id, vector] of Object.entries(rows)) {
    assert.equal(vector.length, width);
    table.set(vector, Number(id) * width);
  }
  const tokenIds = encode(sequence);
  assert.deepEqual(Array.from(tokenIds), ids);
  const output = gatherEmbeddings(tokenIds, table, width);
  assert.equal(output.length, 64);
  const bytes = Buffer.alloc(output.byteLength);
  output.forEach((value, i) => bytes.writeFloatLE(value, i * 4));
  assert.equal(createHash('sha256').update(bytes).digest('hex'), output_float32_le_sha256);
});
