import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createDyna1CPU } from '../cpu/dyna1/index.js';
import {
  diffStats,
  normalizeBackboneCoords,
  normalizeStructureTokens,
  resolvePredictInputs,
  SEQUENCE_MASK_TOKEN,
  STRUCTURE_MASK_TOKEN,
} from '../shared/dyna1/index.js';

const root = new URL('../', import.meta.url);
const read = async url => {
  const b = await readFile(new URL(url, root));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};

test('structure input normalization and validation', () => {
  const flat = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, NaN, Infinity, 0, 1, 2, 3, 4, 5, 6]);
  const a = normalizeBackboneCoords(flat);
  assert.equal(a.length, 2);
  assert.ok(Number.isNaN(a.coords[9]) && Number.isNaN(a.coords[10]));

  const obj = normalizeBackboneCoords({
    n: new Float32Array([1, 2, 3, 10, 11, 12]),
    ca: new Float32Array([4, 5, 6, 13, 14, 15]),
    c: new Float32Array([7, 8, 9, 16, 17, 18]),
  });
  assert.equal(obj.length, 2);
  assert.deepEqual(Array.from(obj.coords), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]);

  const nested = normalizeBackboneCoords([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]);
  assert.equal(nested.length, 1);
  assert.deepEqual(Array.from(nested.coords), [1, 2, 3, 4, 5, 6, 7, 8, 9]);

  assert.deepEqual(Array.from(normalizeStructureTokens([0, -1, 4095, 4096])), [0, STRUCTURE_MASK_TOKEN, 4095, 4096]);
  assert.throws(() => normalizeStructureTokens([4101]));
  assert.throws(() => normalizeBackboneCoords(new Float32Array(8)));

  const so = resolvePredictInputs(null, { structureTokens: [10, 20, 30] });
  assert.equal(so.mode, 'struct-only');
  assert.equal(so.residues, null);
  assert.deepEqual(Array.from(so.seqTokens), [SEQUENCE_MASK_TOKEN, SEQUENCE_MASK_TOKEN, SEQUENCE_MASK_TOKEN]);

  assert.throws(() => resolvePredictInputs('AC', { structureTokens: [10, 20, 30] }), RangeError);
  assert.throws(() => resolvePredictInputs('ACD', { mode: 'seq+struct' }), TypeError);
  assert.throws(() => resolvePredictInputs(null, { mode: 'seq-only' }), TypeError);
});

test('CPU FP32: structure encoder + seq+struct and struct-only match upstream PyTorch oracle', async () => {
  const meta = JSON.parse(new TextDecoder().decode(await read('fixtures/oracle_dyna1_struct.json')));
  const buf = await read('fixtures/oracle_dyna1_struct.bin');
  const ref = key => {
    const rec = meta.tensors[key];
    assert.ok(rec, `Missing oracle tensor ${key}`);
    const raw = buf.slice(rec.byteOffset, rec.byteOffset + rec.byteLength);
    return rec.dtype === 'I32' ? new Int32Array(raw) : new Float32Array(raw);
  };

  const engine = await createDyna1CPU({ baseUrl: 'weights/fp32', read, preloadStructureEncoder: true });
  const metrics = {};
  try {
    for (const name of ['1UBQ', '1UBQ_missing']) {
      const seq = meta.proteins[name].sequence;
      const coords = ref(`${name}.coords`);
      const refToks = ref(`${name}.structure_tokens`);

      // Standalone encodeStructure check
      const enc = engine.encodeStructure(coords);
      assert.deepEqual(enc.structureTokens, refToks, `${name} structure_tokens mismatch`);
      const preVqStats = diffStats(enc.preVq, ref(`${name}.pre_vq`));
      assert.ok(preVqStats.relRms < 1e-4, `${name} pre_vq: ${JSON.stringify(preVqStats)}`);

      // seq+struct
      const both = engine.predict(seq, { structureCoords: coords, returnIntermediates: true });
      assert.equal(both.mode, 'seq+struct');
      assert.deepEqual(both.structureTokens, refToks);
      const bothLogits = diffStats(both.logits, ref(`${name}.seq_struct.logits`));
      const bothProbs = diffStats(both.probs, ref(`${name}.seq_struct.probs`));
      assert.ok(bothLogits.relRms < 1e-4, `${name} seq+struct logits: ${JSON.stringify(bothLogits)}`);
      assert.ok(bothProbs.relRms < 1e-4, `${name} seq+struct probs: ${JSON.stringify(bothProbs)}`);
      if (name === '1UBQ') {
        const embS = diffStats(both.intermediates.embedding, ref(`${name}.seq_struct.embedding`));
        const h22S = diffStats(both.intermediates.final_embeddings, ref(`${name}.seq_struct.final_embeddings`));
        assert.ok(embS.relRms < 1e-4, `1UBQ seq+struct embedding: ${JSON.stringify(embS)}`);
        assert.ok(h22S.relRms < 1e-4, `1UBQ seq+struct final_embeddings: ${JSON.stringify(h22S)}`);
      }

      // struct-only (no sequence)
      const strOnly = engine.predict(null, { structureCoords: coords, returnIntermediates: true });
      assert.equal(strOnly.mode, 'struct-only');
      assert.equal(strOnly.residues, null);
      assert.deepEqual(strOnly.structureTokens, refToks);
      const strLogits = diffStats(strOnly.logits, ref(`${name}.struct_only.logits`));
      const strProbs = diffStats(strOnly.probs, ref(`${name}.struct_only.probs`));
      assert.ok(strLogits.relRms < 1e-4, `${name} struct-only logits: ${JSON.stringify(strLogits)}`);
      assert.ok(strProbs.relRms < 1e-4, `${name} struct-only probs: ${JSON.stringify(strProbs)}`);

      // precomputed structureTokens path matches bit-for-bit
      const viaToks = engine.predict(seq, { structureTokens: refToks });
      assert.equal(diffStats(viaToks.probs, both.probs).maxAbs, 0);

      // seq-only regression check
      const seqOnly = engine.predict(seq);
      assert.equal(seqOnly.mode, 'seq-only');
      const seqProbs = diffStats(seqOnly.probs, ref(`${name}.seq_only.probs`));
      assert.ok(seqProbs.relRms < 1e-4, `${name} seq-only probs: ${JSON.stringify(seqProbs)}`);

      metrics[name] = {
        tokensExact: `${enc.structureTokens.length}/${refToks.length}`,
        preVq: preVqStats,
        seqStruct: { logits: bothLogits, probs: bothProbs, timingMs: both.timingMs },
        structOnly: { logits: strLogits, probs: strProbs, timingMs: strOnly.timingMs },
        seqOnly: { probs: seqProbs, timingMs: seqOnly.timingMs },
      };
      console.log(name, JSON.stringify({
        tokensExact: metrics[name].tokensExact,
        seqStructProbRelRms: bothProbs.relRms,
        structOnlyProbRelRms: strProbs.relRms,
      }));
    }
  } finally {
    engine.destroy();
  }
  await writeFile(new URL('../fixtures/cpu_struct_metrics.json', import.meta.url), JSON.stringify(metrics, null, 2) + '\n');
});
