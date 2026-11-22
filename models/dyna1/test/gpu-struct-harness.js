import { createDyna1Engine, requestWebGpuDevice } from '../webgpu/index.js';
import { diffStats } from '../shared/dyna1/index.js';

const log = msg => fetch('/log', { method: 'POST', body: msg });

export async function run() {
  const params = new URL(import.meta.url).searchParams;
  const precision = params.get('precision') ?? 'fp32';
  const useF16 = params.get('native') === '1';
  const mode = precision + (useF16 ? '-native' : '-core');

  const { adapter, device, hasF16 } = await requestWebGpuDevice({ useF16 });
  const info = adapter.info;
  const adapterInfo = {
    vendor: info.vendor,
    architecture: info.architecture,
    device: info.device,
    description: info.description,
    isFallbackAdapter: info.isFallbackAdapter,
  };
  if (info.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe/i.test(JSON.stringify(adapterInfo)))
    throw new Error('Hardware GPU required: ' + JSON.stringify(adapterInfo));
  if (useF16 && !hasF16) throw new Error('shader-f16 unavailable for native-path test');
  await log('struct ' + mode + ' ' + JSON.stringify(adapterInfo));

  const errors = [];
  device.addEventListener('uncapturederror', e => errors.push(e.error.message));

  const meta = await (await fetch('/fixtures/oracle_dyna1_struct.json')).json();
  const buf = await (await fetch('/fixtures/oracle_dyna1_struct.bin')).arrayBuffer();
  const ref = key => {
    const r = meta.tensors[key];
    if (!r) throw new Error('Missing oracle tensor: ' + key);
    const b = buf.slice(r.byteOffset, r.byteOffset + r.byteLength);
    return r.dtype === 'I32' ? new Int32Array(b) : new Float32Array(b);
  };

  let structProgressEvents = 0;
  const engine = await createDyna1Engine({
    baseUrl: '/weights',
    precision,
    device,
    useF16,
    cache: 'none',
    onProgress: p => { if (p.phase === 'structure-encoder') structProgressEvents++; },
  });

  const metrics = {};
  try {
    // 1. Verify seq-only does NOT load structure_encoder
    const memMainOnly = engine.memoryUsage();
    const ubqSeq = meta.proteins['1UBQ'].sequence;
    const seqOnlyOut = await engine.predict(ubqSeq);
    if (seqOnlyOut.mode !== 'seq-only') throw new Error('Expected seq-only mode');
    if (structProgressEvents !== 0) throw new Error('seq-only unexpectedly fetched structure_encoder');
    if (engine.memoryUsage().totalBytes !== memMainOnly.totalBytes) throw new Error('seq-only changed live GPU memory');
    const seqOnlyDiff = diffStats(seqOnlyOut.probs, ref('1UBQ.seq_only.probs'));

    // 2. First structure-conditioned call lazily loads structure_encoder
    const ubqCoords = ref('1UBQ.coords');
    const encOut = await engine.encodeStructure(ubqCoords);
    if (structProgressEvents === 0) throw new Error('structure_encoder progress events missing');
    const memWithStruct = engine.memoryUsage();
    if (memWithStruct.totalBytes <= memMainOnly.totalBytes) throw new Error('structure_encoder buffers not accounted in memoryUsage');

    // Verify abort signal and error recovery leave no leaked buffers
    const aborted = new AbortController();
    aborted.abort();
    try { await engine.predict(ubqSeq, { structureCoords: ubqCoords, signal: aborted.signal }); throw new Error('Abort ignored'); }
    catch (e) { if (e.name !== 'AbortError') throw e; }
    try { await engine.predict('ACD', { structureCoords: ubqCoords }); throw new Error('Length mismatch ignored'); }
    catch (e) { if (!(e instanceof RangeError)) throw e; }
    if (engine.memoryUsage().totalBytes !== memWithStruct.totalBytes) throw new Error('Error path leaked buffers');

    // Verify { n, ca, c } coordinate format matches flat Float32Array
    const L0 = ubqSeq.length;
    const nArr = new Float32Array(L0 * 3), caArr = new Float32Array(L0 * 3), cArr = new Float32Array(L0 * 3);
    for (let i = 0; i < L0; i++) {
      for (let d = 0; d < 3; d++) {
        nArr[i * 3 + d] = ubqCoords[i * 9 + d];
        caArr[i * 3 + d] = ubqCoords[i * 9 + 3 + d];
        cArr[i * 3 + d] = ubqCoords[i * 9 + 6 + d];
      }
    }
    const encObj = await engine.encodeStructure({ n: nArr, ca: caArr, c: cArr });
    for (let i = 0; i < L0; i++) if (encObj.structureTokens[i] !== encOut.structureTokens[i]) throw new Error('{n,ca,c} format token mismatch');

    // 3. Full parity checks across 1UBQ, 1OCA, 1RYK, 1UBQ_missing
    const probThresh = precision === 'fp32' ? 1e-4 : precision === 'fp16' ? 5e-4 : 0.12;
    const logitThresh = precision === 'fp32' ? 1e-4 : precision === 'fp16' ? 5e-4 : 0.05;
    const corrThresh = precision === 'int8' ? 0.97 : 0.99999;

    for (const name of ['1UBQ', '1OCA', '1RYK', '1UBQ_missing']) {
      const seq = meta.proteins[name].sequence;
      const coords = ref(`${name}.coords`);
      const refToks = ref(`${name}.structure_tokens`);

      const both = await engine.predict(seq, { structureCoords: coords, returnIntermediates: true });
      if (both.mode !== 'seq+struct') throw new Error(`Expected seq+struct for ${name}`);
      let exact = 0;
      for (let i = 0; i < refToks.length; i++) if (both.structureTokens[i] === refToks[i]) exact++;
      if (exact !== refToks.length) throw new Error(`${mode} ${name} structure_tokens mismatch: ${exact}/${refToks.length}`);

      const preVqS = diffStats(both.intermediates.pre_vq, ref(`${name}.pre_vq`));
      if (preVqS.relRms >= 1e-4) throw new Error(`${mode} ${name} pre_vq: ${JSON.stringify(preVqS)}`);

      const bothLogits = diffStats(both.logits, ref(`${name}.seq_struct.logits`));
      const bothProbs = diffStats(both.probs, ref(`${name}.seq_struct.probs`));
      if (bothLogits.relRms >= logitThresh || bothLogits.correlation < corrThresh)
        throw new Error(`${mode} ${name} seq+struct logits: ${JSON.stringify(bothLogits)}`);
      if (bothProbs.relRms >= probThresh || bothProbs.correlation < corrThresh)
        throw new Error(`${mode} ${name} seq+struct probs: ${JSON.stringify(bothProbs)}`);

      const strOnly = await engine.predict(null, { structureCoords: coords });
      if (strOnly.mode !== 'struct-only' || strOnly.residues !== null) throw new Error(`Expected struct-only for ${name}`);
      for (let i = 0; i < refToks.length; i++) if (strOnly.structureTokens[i] !== refToks[i]) throw new Error(`${mode} ${name} struct-only token mismatch`);
      const strLogits = diffStats(strOnly.logits, ref(`${name}.struct_only.logits`));
      const strProbs = diffStats(strOnly.probs, ref(`${name}.struct_only.probs`));
      if (strLogits.relRms >= logitThresh || strLogits.correlation < corrThresh)
        throw new Error(`${mode} ${name} struct-only logits: ${JSON.stringify(strLogits)}`);
      if (strProbs.relRms >= probThresh || strProbs.correlation < corrThresh)
        throw new Error(`${mode} ${name} struct-only probs: ${JSON.stringify(strProbs)}`);

      // Warm timings (3 runs)
      const warmStructMs = [];
      const warmEncMs = [];
      for (let i = 0; i < 3; i++) {
        warmEncMs.push((await engine.encodeStructure(coords)).timingMs);
        warmStructMs.push((await engine.predict(seq, { structureCoords: coords })).timingMs);
      }
      if (engine.memoryUsage().totalBytes !== memWithStruct.totalBytes) throw new Error(`Activation buffers leaked on ${name}`);

      metrics[name] = {
        length: refToks.length,
        tokensExact: exact,
        tokensTotal: refToks.length,
        preVq: preVqS,
        seqStruct: {
          logits: bothLogits,
          probs: bothProbs,
          medianMs: [...warmStructMs].sort((a, b) => a - b)[1],
          warmMs: warmStructMs,
          encodeMedianMs: [...warmEncMs].sort((a, b) => a - b)[1],
        },
        structOnly: { logits: strLogits, probs: strProbs, coldMs: strOnly.timingMs },
      };
      await log(`${mode} ${name} tokens=${exact}/${refToks.length} seq+struct_prob_relRms=${bothProbs.relRms.toExponential(3)} r=${bothProbs.correlation.toFixed(6)} struct-only_prob_relRms=${strProbs.relRms.toExponential(3)} encMs=${metrics[name].seqStruct.encodeMedianMs.toFixed(1)} totalMs=${metrics[name].seqStruct.medianMs.toFixed(1)}`);
    }

    // 4. Verify multi-chunk structure encoding (> 192 residues, e.g. L = 228 = 3 x 1UBQ shifted)
    const Lchunk = 76 * 3;
    const chunkCoords = new Float32Array(Lchunk * 9);
    for (let rep = 0; rep < 3; rep++) {
      for (let i = 0; i < 76; i++) {
        for (let a = 0; a < 3; a++) {
          chunkCoords[(rep * 76 + i) * 9 + a * 3 + 0] = ubqCoords[i * 9 + a * 3 + 0] + rep * 100;
          chunkCoords[(rep * 76 + i) * 9 + a * 3 + 1] = ubqCoords[i * 9 + a * 3 + 1];
          chunkCoords[(rep * 76 + i) * 9 + a * 3 + 2] = ubqCoords[i * 9 + a * 3 + 2];
        }
      }
    }
    const chunkEnc = await engine.encodeStructure(chunkCoords);
    if (chunkEnc.structureTokens.length !== Lchunk) throw new Error('Chunked encodeStructure length mismatch');
    const refUbqToks = ref('1UBQ.structure_tokens');
    // Each translated copy is 100 Å away (> any intra-ubiquitin distance), so each copy's 16-NN neighborhood is identical to 1UBQ!
    let chunkExact = 0;
    for (let rep = 0; rep < 3; rep++) for (let i = 0; i < 76; i++) if (chunkEnc.structureTokens[rep * 76 + i] === refUbqToks[i]) chunkExact++;
    if (chunkExact !== Lchunk) throw new Error(`Chunked L=228 token mismatch: ${chunkExact}/${Lchunk}`);
    if (engine.memoryUsage().totalBytes !== memWithStruct.totalBytes) throw new Error('Chunked encodeStructure leaked buffers');

    metrics._summary = {
      mode,
      mainWeightBytes: memMainOnly.totalBytes,
      withStructWeightBytes: memWithStruct.totalBytes,
      structEncoderWeightBytes: memWithStruct.totalBytes - memMainOnly.totalBytes,
      seqOnlyUbqProbRelRms: seqOnlyDiff.relRms,
      chunked228TokensExact: `${chunkExact}/${Lchunk}`,
      chunked228EncodeMs: chunkEnc.timingMs,
    };
  } finally {
    engine.destroy();
    engine.destroy();
    if (engine.memoryUsage().bufferCount !== 0) throw new Error('destroy leaked GPU buffers');
  }
  device.destroy();
  if (errors.length) throw new Error(errors.join('\n'));
  return { adapterInfo, hasF16, mode, metrics, validationErrors: errors };
}
