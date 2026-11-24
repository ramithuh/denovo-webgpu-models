import { createLaProteinaEngine } from '../webgpu/index.js';
import { createLaProteinaWorker } from '../webgpu/worker-client.js';
import { acquireAdapter } from '../webgpu/device.js';
import { assert, fixture } from './parity.js';
import {
  ae3InputFromRef,
  checkAe3Output,
  verifyPdbMotifExtraction,
  motifSampleOptions,
  checkMotifFinalOutput,
  checkMotifOutput,
} from './motif-parity.js';

const read = async (p) => {
  const r = await fetch('/' + p);
  if (!r.ok) throw Error(`Missing fixture ${p}`);
  return r.arrayBuffer();
};
const readText = async (p) => {
  const r = await fetch('/' + p);
  if (!r.ok) throw Error(`Missing text ${p}`);
  return r.text();
};

export async function run() {
  assert(navigator.gpu, 'Hardware WebGPU is required');
  const shaders = new Set();
  // Preserve any existing shaders in .cache/shaders/manifest.json so validate_wgsl.py audits the full union!
  try {
    const prevManifestResp = await fetch('/.cache/shaders/manifest.json');
    if (prevManifestResp.ok) {
      const prevFiles = await prevManifestResp.json();
      for (const f of prevFiles) {
        const sr = await fetch('/.cache/shaders/' + f);
        if (sr.ok) shaders.add(await sr.text());
      }
    }
  } catch {}

  const createShader = GPUDevice.prototype.createShaderModule;
  GPUDevice.prototype.createShaderModule = function (descriptor) {
    shaders.add(descriptor.code);
    return createShader.call(this, descriptor);
  };

  const manifest = await (await fetch('/.cache/fixtures/motif_oracles.json')).json();
  const extManifest = await (await fetch('/.cache/fixtures/motif_extended_oracles.json')).json();
  const longManifest = await (await fetch('/.cache/fixtures/motif_long_oracles.json')).json();
  const longOdeManifest = await (await fetch('/.cache/fixtures/motif_long_ode_oracles.json')).json();

  const pdbCache = new Map();
  const getPdb = async (rel) => {
    if (!pdbCache.has(rel)) pdbCache.set(rel, await readText(`.cache/upstream/${rel}`));
    return pdbCache.get(rel);
  };

  const ae3Results = [];
  const ldResults = [];
  const longResults = [];
  const memory = {};
  const variantMemory = {};
  const lifecycle = {};
  const benchmarks = {};
  let adapterInfo = null;

  // 1. Per-variant VRAM packing check across fp32, fp16, int8 (verifying <0.51x and <0.26x & zero FP32 weight shadows)
  for (const precision of ['fp32', 'fp16', 'int8']) {
    const loadStart = performance.now();
    const engine = await createLaProteinaEngine({
      baseUrl: location.origin + '/weights',
      precision,
      variants: ['ae3', 'ld4', 'ld5', 'ld6', 'ld7'],
      allowSoftwareFallback: false,
    });
    adapterInfo = engine.adapterInfo;
    assert(!engine.isFallbackAdapter, 'Fallback adapter forbidden');
    memory[precision] = { loaded: engine.memoryUsage(), loadMs: performance.now() - loadStart };
    await fetch('/log', {
      method: 'POST',
      body: `Loaded all_motif (${precision}) in ${memory[precision].loadMs.toFixed(1)} ms (${(memory[precision].loaded.bytes / 1e6).toFixed(1)} MB)`,
    });

    try {
      // A. AE3 encoder -> decoder round-trip cases
      for (const c of manifest.ae3_cases.filter((x) => x.precision === precision)) {
        const ref = await fixture(c, read);
        const wantTrace = precision === 'fp32' && Boolean(ref['trace.enc.init_repr_factory']);
        const out = await engine.encodeDecode(ae3InputFromRef(c, ref), { returnIntermediates: wantTrace });
        const parity = checkAe3Output(out, ref, c);
        assert(engine.memoryUsage().transientBuffers === 0, `Transient leak after ${c.name}`);
        ae3Results.push({
          name: c.name,
          precision,
          length: c.length,
          timingMs: out.timingMs,
          reconstructionRmsdAng: c.reconstruction_rmsd_ang,
          seqRecovery: c.seq_recovery,
          parity,
        });
        await fetch('/log', {
          method: 'POST',
          body: `PASS ${c.name} ${out.timingMs.toFixed(1)} ms (coords relRMS=${parity.coords.relRMS.toExponential(2)})`,
        });
      }

      // B. Short + Extended dynamic-shape LD4..LD7 motif cases (all 4 models, L=26, 27, 52, T=1, 2, 3, sde + ode)
      const allShortCases = [
        ...manifest.cases.filter((x) => x.precision === precision),
        ...extManifest.cases.filter((x) => x.precision === precision),
      ];
      for (const c of allShortCases) {
        const ref = await fixture(c, read);
        const pdbText = await getPdb(c.pdb);
        if (precision === 'fp32') verifyPdbMotifExtraction(pdbText, c, ref);
        const out = await engine.sample(motifSampleOptions(c, ref, pdbText));
        const parity = checkMotifOutput(out, ref, c);
        assert(engine.memoryUsage().transientBuffers === 0, `Transient leak after ${c.name}`);
        ldResults.push({
          name: c.name,
          model: c.model,
          precision,
          task: c.task,
          length: c.length,
          steps: c.steps,
          mode: c.mode,
          timingMs: out.timingMs,
          motifMetrics: out.motifMetrics,
          parity,
        });
        await fetch('/log', {
          method: 'POST',
          body: `PASS ${c.name} ${out.timingMs.toFixed(1)} ms (coords relRMS=${parity.coords.relRMS.toExponential(2)})`,
        });
      }

      // C1. Full 400-step trajectory + final parity (relRMS < 1e-4 across all 400 steps) on LD4 SDE + ODE, plus 400-step final parity on LD5 ODE
      const traj400Cases = [
        ...longOdeManifest.cases.filter((x) => x.precision === precision && (x.model === 'LD4' || x.model === 'LD5')),
        ...longManifest.cases.filter((x) => x.precision === precision && x.model === 'LD4'),
      ];
      for (const c of traj400Cases) {
        await fetch('/log', { method: 'POST', body: `Starting 400-step parity ${c.name}...` });
        const ref = await fixture(c, read);
        const pdbText = await getPdb(c.pdb);
        const fullTraj = c.model === 'LD4';
        const out = await engine.sample({ ...motifSampleOptions(c, ref, pdbText), returnTrajectory: fullTraj });
        const parity = fullTraj ? checkMotifOutput(out, ref, c) : checkMotifFinalOutput(out, ref, c);
        assert(engine.memoryUsage().transientBuffers === 0, `Transient leak after 400-step ${c.name}`);
        longResults.push({
          name: c.name,
          model: c.model,
          precision,
          mode: c.mode,
          task: c.task,
          length: c.length,
          steps: c.steps,
          timingMs: out.timingMs,
          msPerStep: out.timingMs / c.steps,
          sequence: out.sequence,
          motifMetrics: out.motifMetrics,
          fullTrajectoryParity: fullTraj,
          coordsRelRMS: parity.coords.relRMS,
        });
        await fetch('/log', {
          method: 'POST',
          body: `PASS 400-step ${c.name} in ${out.timingMs.toFixed(1)} ms (${(out.timingMs / 400).toFixed(2)} ms/step, coords relRMS=${parity.coords.relRMS.toExponential(2)}, motifRMSD=${out.motifMetrics.motifRmsdAlignedAngstrom.toFixed(3)}A)`,
        });
      }

      // C2. Full 400-step SDE atomistic motif preservation across all 4 models (LD4..LD7): motifRMSD < 0.25 A & 100% motif seq recovery
      for (const c of longManifest.cases.filter((x) => x.precision === precision && x.model !== 'LD4')) {
        await fetch('/log', { method: 'POST', body: `Starting 400-step SDE motif RMSD check ${c.name}...` });
        const ref = await fixture(c, read);
        const pdbText = await getPdb(c.pdb);
        const out = await engine.sample({ ...motifSampleOptions(c, ref, pdbText), returnTrajectory: false });
        assert(out.motifMetrics.motifRmsdAlignedAngstrom < 0.25, `${c.name}: motif RMSD ${out.motifMetrics.motifRmsdAlignedAngstrom} >= 0.25 A`);
        assert(out.motifMetrics.motifSeqRecovery === 1.0, `${c.name}: motif seq recovery ${out.motifMetrics.motifSeqRecovery} !== 1.0`);
        assert(engine.memoryUsage().transientBuffers === 0, `Transient leak after 400-step SDE ${c.name}`);
        longResults.push({
          name: c.name,
          model: c.model,
          precision,
          mode: c.mode,
          task: c.task,
          length: c.length,
          steps: c.steps,
          timingMs: out.timingMs,
          msPerStep: out.timingMs / c.steps,
          sequence: out.sequence,
          motifMetrics: out.motifMetrics,
          pyMotifRmsdAng: c.motif_rmsd_aligned_ang,
          pyMotifSeqRecovery: c.motif_seq_recovery,
        });
        await fetch('/log', {
          method: 'POST',
          body: `PASS 400-step SDE motif preservation ${c.name} in ${out.timingMs.toFixed(1)} ms (${(out.timingMs / 400).toFixed(2)} ms/step, motifRMSD=${out.motifMetrics.motifRmsdAlignedAngstrom.toFixed(3)}A [py=${c.motif_rmsd_aligned_ang.toFixed(3)}A], seqRec=${(out.motifMetrics.motifSeqRecovery * 100).toFixed(0)}%)`,
        });
      }

      // D. Warm GB10 Benchmarks (AE3 encode/decode/encodeDecode + LD4..LD7 L=100 T=400 / step timings on int8 & fp16)
      if (precision === 'int8' || precision === 'fp16') {
        const pdb1prw = await getPdb('motif_benchmark_pdb_files/1prw_aa.pdb');
        const pdb1ycr = await getPdb('motif_benchmark_pdb_files/1ycr_aa.pdb');
        // Warmup + measure AE3 encodeDecode on 1PRW (L=40)
        await engine.encodeDecode({ pdb: pdb1prw, seed: 7 });
        const ae3Warm = await engine.encodeDecode({ pdb: pdb1prw, seed: 7 });
        const encWarm = await engine.encode({ pdb: pdb1prw, seed: 7 });

        const modelStepMs = {};
        for (const v of ['ld4', 'ld5', 'ld6', 'ld7']) {
          const cond100 = { pdb: pdb1ycr, contig: '45/B19-27/46', motifOnly: true }; // L = 100 (11 motif res + 89 scaffold)
          await engine.sample({ variant: v, steps: 2, seed: 11, motif: cond100 });
          const t10 = await engine.sample({ variant: v, steps: 10, seed: 11, motif: cond100 });
          const t2 = await engine.sample({ variant: v, steps: 2, seed: 11, motif: cond100 });
          modelStepMs[v] = {
            L100_msPerStep: (t10.timingMs - t2.timingMs) / 8,
            L100_estimated400StepsMs: ((t10.timingMs - t2.timingMs) / 8) * 400 + encWarm.timingMs,
          };
        }
        if (precision === 'int8') {
          // Run actual full L=100, T=400 SDE warm benchmark for LD4 (indexed all-atom) and LD6 (unindexed all-atom)!
          const cond100 = { pdb: pdb1ycr, contig: '45/B19-27/46', motifOnly: true };
          const ld4Full100 = await engine.sample({ variant: 'ld4', steps: 400, seed: 5, motif: cond100 });
          const ld6Full100 = await engine.sample({ variant: 'ld6', steps: 400, seed: 5, motif: cond100 });
          modelStepMs.ld4.L100_T400_actualMs = ld4Full100.timingMs;
          modelStepMs.ld4.L100_T400_motifMetrics = ld4Full100.motifMetrics;
          modelStepMs.ld6.L100_T400_actualMs = ld6Full100.timingMs;
          modelStepMs.ld6.L100_T400_motifMetrics = ld6Full100.motifMetrics;
          await fetch('/log', {
            method: 'POST',
            body: `BENCH L=100 T=400 INT8: LD4=${ld4Full100.timingMs.toFixed(1)}ms (motifRMSD=${ld4Full100.motifMetrics.motifRmsdAlignedAngstrom.toFixed(3)}A), LD6=${ld6Full100.timingMs.toFixed(1)}ms (motifRMSD=${ld6Full100.motifMetrics.motifRmsdAlignedAngstrom.toFixed(3)}A)`,
          });
        }
        benchmarks[precision] = {
          ae3_L40_encodeMs: encWarm.timingMs,
          ae3_L40_encodeDecodeMs: ae3Warm.timingMs,
          models: modelStepMs,
        };
      }

      // E. Lifecycle: error & abort cleanup on motif + AE3
      let errs = 0;
      try {
        await engine.sample({ variant: 'ld4', steps: 1 }); // missing motif
      } catch {
        errs++;
      }
      assert(errs === 1 && engine.memoryUsage().transientBuffers === 0, 'Missing-motif error leaked buffers');
      const ctrl = new AbortController();
      const pdb1ycr = await getPdb('motif_benchmark_pdb_files/1ycr_aa.pdb');
      try {
        await engine.sample({
          variant: 'ld6',
          steps: 3,
          motif: { pdb: pdb1ycr, contig: '8/B19-27/9', motifOnly: true },
          signal: ctrl.signal,
          onProgress: (p) => {
            if (p.phase === 'ld6') ctrl.abort();
          },
        });
      } catch (e) {
        assert(e.name === 'AbortError', 'Expected AbortError');
        errs++;
      }
      assert(errs === 2 && engine.memoryUsage().transientBuffers === 0, 'Motif abort leaked buffers');
      lifecycle[precision] = { errorAndAbortClean: true, ...engine.memoryUsage() };
    } finally {
      engine.destroy();
      engine.destroy();
    }
    assert(engine.memoryUsage().bytes === 0, `Motif GPU destroy leaked bytes (${precision})`);
  }

  // Verify packed VRAM ratios across the combined motif bundle
  assert(memory.fp16.loaded.bytes / memory.fp32.loaded.bytes < 0.51, 'Motif FP16 weights not packed');
  assert(memory.int8.loaded.bytes / memory.fp32.loaded.bytes < 0.26, 'Motif INT8 weights not packed');

  // Verify single-variant VRAM (e.g. variant: 'ld4' loads LD4 + AE3 only, not LD5..LD7)
  const singleLd4 = await createLaProteinaEngine({
    baseUrl: location.origin + '/weights',
    precision: 'int8',
    variant: 'ld4',
    allowSoftwareFallback: false,
  });
  variantMemory.ld4_int8_bytes = singleLd4.memoryUsage().bytes;
  singleLd4.destroy();
  assert(variantMemory.ld4_int8_bytes < memory.int8.loaded.bytes * 0.5, 'Single motif variant loaded unnecessary bundles');

  // 2. Native shader-f16 (useF16: true) check on AE3 + LD4 + LD6
  const { adapter } = await acquireAdapter({ allowSoftwareFallback: false });
  assert(adapter.features.has('shader-f16'), 'shader-f16 required on GB10');
  const device = await adapter.requestDevice({
    requiredFeatures: ['shader-f16'],
    requiredLimits: {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  });
  let nativeF16 = {};
  try {
    const native = await createLaProteinaEngine({
      baseUrl: location.origin + '/weights',
      precision: 'fp16',
      variants: ['ae3', 'ld4', 'ld6'],
      device,
      adapter,
      useF16: true,
      allowSoftwareFallback: false,
    });
    try {
      assert(native.usesShaderF16, 'usesShaderF16 must be true');
      const aeCase = manifest.ae3_cases.find((x) => x.precision === 'fp16');
      const aeRef = await fixture(aeCase, read);
      const aeOut = await native.encodeDecode(ae3InputFromRef(aeCase, aeRef));
      const ld4Case = manifest.cases.find((x) => x.precision === 'fp16' && x.model === 'LD4' && x.steps === 3 && x.mode === 'sde');
      const ld4Ref = await fixture(ld4Case, read);
      const ld4Out = await native.sample(motifSampleOptions(ld4Case, ld4Ref, await getPdb(ld4Case.pdb)));
      const ld6Case = manifest.cases.find((x) => x.precision === 'fp16' && x.model === 'LD6' && x.steps === 3 && x.mode === 'sde');
      const ld6Ref = await fixture(ld6Case, read);
      const ld6Out = await native.sample(motifSampleOptions(ld6Case, ld6Ref, await getPdb(ld6Case.pdb)));
      nativeF16 = {
        ae3: checkAe3Output(aeOut, aeRef, aeCase),
        ld4: checkMotifOutput(ld4Out, ld4Ref, ld4Case),
        ld6: checkMotifOutput(ld6Out, ld6Ref, ld6Case),
        packedBytes: native.memoryUsage().bytes,
      };
    } finally {
      native.destroy();
    }
  } finally {
    device.destroy();
  }

  // 3. Web Worker check with variant: 'ld5' (indexed tip-atom + AE3 encodeDecode)
  const worker = await createLaProteinaWorker({
    baseUrl: location.origin + '/weights',
    precision: 'int8',
    variant: 'ld5',
    allowSoftwareFallback: false,
  });
  let workerResult;
  try {
    const c = manifest.cases.find((x) => x.precision === 'int8' && x.model === 'LD5' && x.task === '1prw_L52' && x.steps === 3 && x.mode === 'sde');
    const ref = await fixture(c, read);
    const out = await worker.sample(motifSampleOptions(c, ref, await getPdb(c.pdb)));
    const aeC = manifest.ae3_cases.find((x) => x.precision === 'int8' && x.length === 40);
    const aeRef = await fixture(aeC, read);
    const aeOut = await worker.encodeDecode(ae3InputFromRef(aeC, aeRef));
    const mem = await worker.memoryUsage();
    assert(mem.transientBuffers === 0, 'Worker motif transient leak');
    workerResult = {
      variant: worker.variant,
      ld5Parity: checkMotifOutput(out, ref, c),
      ae3Parity: checkAe3Output(aeOut, aeRef, aeC),
      memory: mem,
    };
  } finally {
    const m = await worker.destroy();
    assert(m.bytes === 0, 'Worker motif destroy leak');
  }

  const saved = await fetch('/shaders', { method: 'POST', body: JSON.stringify([...shaders]) });
  assert(saved.ok, 'Motif shader audit save failed');
  GPUDevice.prototype.createShaderModule = createShader;

  return {
    adapterInfo,
    ae3Cases: ae3Results,
    cases: ldResults,
    longCases: longResults,
    memory,
    variantMemory,
    benchmarks,
    nativeF16,
    worker: workerResult,
    lifecycle,
    shaderVariants: shaders.size,
    skipped: 0,
  };
}
