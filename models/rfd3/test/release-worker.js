import { createRfd3Engine } from '../webgpu/index.js';
import { buildInput, structureFromResult } from '../shared/input.js';

async function run() {
  const architecture = await (await fetch('/fixtures/architecture.json')).json();
  const triad = await (await fetch('/fixtures/triad.pdb')).text();
  const conditioning = await (await fetch('/fixtures/conditioning.pdb')).text();

  // P2-6: Create engine with baseURL WITHOUT trailing slash, auto-loaded manifest, custom read(), and onProgress()
  const baseNoSlash = new URL('/weights/fp16', location.href).href;
  const progressEvents = [];
  const readUrls = [];
  const engine = await createRfd3Engine({
    architecture,
    baseURL: baseNoSlash,
    cache: 'none',
    read: async (url, { signal } = {}) => {
      readUrls.push(url);
      const res = await fetch(url, { signal, cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${url}`);
      return res.arrayBuffer();
    },
    onProgress: ev => progressEvents.push(ev),
  });

  try {
    const baselineBytes = engine.memoryUsage();
    const out = {
      baselineBytes,
      progressCount: progressEvents.length,
      progressLoaded: progressEvents.at(-1)?.loaded,
      progressTotal: progressEvents.at(-1)?.total,
      readUrlCount: readUrls.length,
      readUrlHasFp16: readUrls.every(u => u.includes('/weights/fp16/')),
    };

    // 1. P1-5: Measure peak VRAM at L=160 (was 7.02 GB before!)
    engine.trim();
    const input160 = buildInput({ length: 160 });
    const t0 = performance.now();
    const handle160 = await engine.initialize(input160.features);
    const init160Ms = performance.now() - t0;
    const init160PeakBytes = engine.peakMemoryUsage();
    const init160ResidentBytes = engine.memoryUsage();
    await engine.run({
      features: input160.features,
      X_noisy_L: new Float32Array(input160.features.atom_to_token_map.length * 3),
      t: 16,
      n_recycle: 2,
      initializer: handle160,
    });
    const full160PeakBytes = engine.peakMemoryUsage();
    engine.releaseInitializer(handle160);
    engine.trim();
    const after160ResidentBytes = engine.memoryUsage();
    out.l160 = {
      init160Ms,
      init160PeakBytes,
      init160PeakGB: Number((init160PeakBytes / 1e9).toFixed(3)),
      init160ResidentBytes,
      full160PeakBytes,
      full160PeakGB: Number((full160PeakBytes / 1e9).toFixed(3)),
      leakedBytesAfter160: after160ResidentBytes - baselineBytes,
    };

    // 2. P1-5: Recovery after RangeError in generate() AND mid-transaction GPU buffer rollback inside transaction()
    let oversizedErr = null;
    try {
      await engine.generate({ length: 1000, steps: 2 });
    } catch (e) {
      oversizedErr = { name: e.name, message: e.message };
    }
    const residentAfterOversized = engine.memoryUsage();

    // Trigger a genuine mid-transaction RangeError AFTER a GPU buffer is already uploaded inside transaction():
    const hSmall = await engine.initialize(buildInput({ length: 4 }).features);
    const residentWithHandle = engine.memoryUsage();
    let txErr = null;
    try {
      // 40,000 atoms = 480 KB upload succeeds inside transaction(), then l*l (1.6B floats = 6.4 GB) throws RangeError mid-transaction!
      await engine.inspectNeighbours(hSmall, new Float32Array(40_000 * 3));
    } catch (e) {
      txErr = { name: e.name, message: e.message };
    }
    const residentAfterTxErr = engine.memoryUsage();
    engine.releaseInitializer(hSmall);
    engine.trim();
    const residentAfterRelease = engine.memoryUsage();

    // Immediately run a valid generate() on the SAME engine (proving engine recovered cleanly after mid-transaction RangeError)
    const recovery = await engine.generate({ length: 16, steps: 10, seed: 7, retainTrajectory: false });
    // Also verify the exact P1-4 reproducer (L=16, steps=4, seed=7) decodes nucleotide argmaxes without throwing
    const p14Input = buildInput({ length: 16 });
    const p14Sample = await engine.sample({ features: p14Input.features, steps: 4, seed: 7, retainTrajectory: false });
    const p14NonProteinCount = Array.from(p14Sample.sequence_indices_I.data).filter(c => c >= 20).length;
    // Scale coordinates into PDB range only if extreme low-step overshoot occurred, just to verify structureFromResult decoding never throws on nucleotide argmax
    const maxCoord = p14Sample.X_L.data.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
    const p14Coords = maxCoord > 900 ? Float32Array.from(p14Sample.X_L.data, v => v * (100 / maxCoord)) : p14Sample.X_L.data;
    const p14Struct = structureFromResult(p14Input, { ...p14Sample, X_L: { ...p14Sample.X_L, data: p14Coords } });
    engine.trim();
    const residentAfterRecovery = engine.memoryUsage();
    out.recovery = {
      oversizedErr,
      txErr,
      leakedAfterOversized: residentAfterOversized - baselineBytes,
      leakedMidTransaction: residentAfterTxErr - residentWithHandle,
      leakedAfterRelease: residentAfterRelease - baselineBytes,
      leakedAfterRecovery: residentAfterRecovery - baselineBytes,
      recoveredRows: recovery.rows.length,
      recoveredFinite: recovery.X_L.data.every(Number.isFinite),
      p1_4_nonProteinArgmaxSeen: p14NonProteinCount,
      p1_4_lowStepL16Seed7Passed: p14Struct.rows.length > 0 && p14Struct.pdb.includes('ATOM'),
    };

    // 4. P1-1: triad.pdb unindexed catalytic motif end-to-end on GPU
    const triadOut = await engine.generate({
      mode: 'motif',
      length: 24,
      structure: triad,
      unindex: 'A1,A2,A3',
      fixedAtoms: { A1: 'ND1,NE2', A2: 'OG', A3: 'OD1,OD2' },
      steps: 11,
      seed: 42,
      retainTrajectory: false,
    });
    out.triad = {
      finite: triadOut.X_L.data.every(Number.isFinite),
      assignments: triadOut.motifAssignments,
      hasHIS: triadOut.rows.some(r => r.label_comp_id === 'HIS'),
      hasSER: triadOut.rows.some(r => r.label_comp_id === 'SER'),
      hasASP: triadOut.rows.some(r => r.label_comp_id === 'ASP'),
    };

    // 5. P1-2: non-ATP multi-atom PDB ligand + canonical ZN component end-to-end on GPU
    const ligPdb = [
      'HETATM    1  C1  LIG L   1       0.000   0.000   0.000  1.00  0.00           C',
      'HETATM    2  C2  LIG L   1       1.520   0.000   0.000  1.00  0.00           C',
      'HETATM    3  O1  LIG L   1       2.200   1.200   0.000  1.00  0.00           O',
      'END',
    ].join('\n');
    const ligZnOut = await engine.generate({
      mode: 'conditioned',
      length: 12,
      structure: ligPdb,
      ligand: 'LIG',
      components: [{ name: 'ZN', translation: [4, 4, 4] }],
      steps: 11,
      seed: 123,
      retainTrajectory: false,
    });
    out.ligandZn = {
      finite: ligZnOut.X_L.data.every(Number.isFinite),
      hasLig: ligZnOut.rows.some(r => r.label_comp_id === 'LIG'),
      hasZn: ligZnOut.rows.some(r => r.label_comp_id === 'ZN'),
      originFinite: ligZnOut.origin.every(Number.isFinite),
    };

    // 6. P1-3: partialT with fixed target + diffused unfixed-sequence chain end-to-end on GPU
    const partialPdb = [
      'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N',
      'ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C',
      'ATOM      3  C   ALA A   1       2.009   1.421   0.000  1.00  0.00           C',
      'ATOM      4  O   ALA A   1       1.355   2.428   0.000  1.00  0.00           O',
      'ATOM      5  CB  ALA A   1       1.988  -0.773  -1.208  1.00  0.00           C',
      'ATOM      6  N   GLY A   2       3.330   1.510   0.000  1.00  0.00           N',
      'ATOM      7  CA  GLY A   2       4.000   2.800   0.000  1.00  0.00           C',
      'ATOM      8  C   GLY A   2       5.500   2.600   0.000  1.00  0.00           C',
      'ATOM      9  O   GLY A   2       6.000   1.500   0.000  1.00  0.00           O',
      'ATOM     10  N   ALA B   1      20.000   0.000   0.000  1.00  0.00           N',
      'ATOM     11  CA  ALA B   1      21.458   0.000   0.000  1.00  0.00           C',
      'ATOM     12  C   ALA B   1      22.009   1.421   0.000  1.00  0.00           C',
      'ATOM     13  O   ALA B   1      21.355   2.428   0.000  1.00  0.00           O',
      'ATOM     14  CB  ALA B   1      21.988  -0.773  -1.208  1.00  0.00           C',
      'END',
    ].join('\n');
    const partialOut = await engine.generate({
      mode: 'motif',
      structure: partialPdb,
      contig: 'A1-2,/0,B1',
      partialT: 5,
      fixedAtoms: { B1: 'ALL' },
      unfixedSequence: 'A1-2',
      steps: 21,
      seed: 99,
      retainTrajectory: false,
    });
    out.partial = {
      finite: partialOut.X_L.data.every(Number.isFinite),
      originX: partialOut.origin[0],
      refPlddt: Array.from(partialOut.features.ref_plddt),
      rows: partialOut.rows.length,
    };

    // 7. P2-3: High-level generate() with cfgScale: 1.5
    const cfgOut = await engine.generate({
      mode: 'motif',
      structure: conditioning,
      contig: '6,A1,6',
      fixedAtoms: { A1: 'N,CA' },
      hbondDonor: { A1: 'N' },
      cfgScale: 1.5,
      steps: 11,
      seed: 77,
      retainTrajectory: false,
    });
    out.cfg = {
      finite: cfgOut.X_L.data.every(Number.isFinite),
      rows: cfgOut.rows.length,
      bFactorsFinite: cfgOut.rows.every(r => Number.isFinite(Number(r.b_iso_or_equiv)) && Number(r.b_iso_or_equiv) >= 0),
      maxBFactor: cfgOut.rows.reduce((m, r) => Math.max(m, Number(r.b_iso_or_equiv)), 0),
    };

    return out;
  } finally {
    engine.destroy();
  }
}

run().then(postMessage, error => postMessage({ error: error.stack ?? String(error) }));
