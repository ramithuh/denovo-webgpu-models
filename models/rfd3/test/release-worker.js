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

    // 8. Issue #98: 1CTT-scale benchmark (L=184 + 4 unindexed catalytic residues + ZN + 16-atom DHZ => n=205, l=2603)
    engine.trim();
    const theozyme1ctt = [
      'ATOM      1  N   HIS A 102       0.000   0.000   0.000  1.00 20.00           N',
      'ATOM      2  CA  HIS A 102       1.458   0.000   0.000  1.00 20.00           C',
      'ATOM      3  C   HIS A 102       2.009   1.420   0.000  1.00 20.00           C',
      'ATOM      4  O   HIS A 102       1.251   2.390   0.000  1.00 20.00           O',
      'ATOM      5  CB  HIS A 102       1.988  -0.773  -1.208  1.00 20.00           C',
      'ATOM      6  CG  HIS A 102       3.450  -0.950  -1.250  1.00 20.00           C',
      'ATOM      7  ND1 HIS A 102       4.250  -0.100  -1.950  1.00 20.00           N',
      'ATOM      8  CD2 HIS A 102       4.220  -1.880  -0.650  1.00 20.00           C',
      'ATOM      9  CE1 HIS A 102       5.500  -0.500  -1.780  1.00 20.00           C',
      'ATOM     10  NE2 HIS A 102       5.520  -1.580  -0.990  1.00 20.00           N',
      'ATOM     11  N   GLU A 104       6.000   2.000   1.000  1.00 20.00           N',
      'ATOM     12  CA  GLU A 104       7.458   2.000   1.000  1.00 20.00           C',
      'ATOM     13  C   GLU A 104       8.009   3.420   1.000  1.00 20.00           C',
      'ATOM     14  O   GLU A 104       7.251   4.390   1.000  1.00 20.00           O',
      'ATOM     15  CB  GLU A 104       7.988   1.227  -0.208  1.00 20.00           C',
      'ATOM     16  CG  GLU A 104       7.600  -0.240  -0.250  1.00 20.00           C',
      'ATOM     17  CD  GLU A 104       6.120  -0.520  -0.300  1.00 20.00           C',
      'ATOM     18  OE1 GLU A 104       5.300   0.400  -0.250  1.00 20.00           O',
      'ATOM     19  OE2 GLU A 104       5.780  -1.710  -0.400  1.00 20.00           O',
      'ATOM     20  N   CYS A 129       2.000  -4.000  -2.000  1.00 20.00           N',
      'ATOM     21  CA  CYS A 129       3.458  -4.000  -2.000  1.00 20.00           C',
      'ATOM     22  C   CYS A 129       4.009  -2.580  -2.000  1.00 20.00           C',
      'ATOM     23  O   CYS A 129       3.251  -1.610  -2.000  1.00 20.00           O',
      'ATOM     24  CB  CYS A 129       3.988  -4.773  -3.208  1.00 20.00           C',
      'ATOM     25  SG  CYS A 129       3.800  -3.850  -4.750  1.00 20.00           S',
      'ATOM     26  N   CYS A 132       6.500  -4.000  -4.000  1.00 20.00           N',
      'ATOM     27  CA  CYS A 132       7.958  -4.000  -4.000  1.00 20.00           C',
      'ATOM     28  C   CYS A 132       8.509  -2.580  -4.000  1.00 20.00           C',
      'ATOM     29  O   CYS A 132       7.751  -1.610  -4.000  1.00 20.00           O',
      'ATOM     30  CB  CYS A 132       8.488  -4.773  -5.208  1.00 20.00           C',
      'ATOM     31  SG  CYS A 132       6.950  -4.900  -6.200  1.00 20.00           S',
      'HETATM   32 ZN    ZN B 296       4.900  -2.200  -3.600  1.00 15.00          ZN2+',
      ...Array.from({ length: 16 }, (_, k) => {
        const el = k < 10 ? 'C' : k < 13 ? 'N' : 'O';
        const nm = `${el}${k + 1}`.padEnd(3);
        const x = (3.5 + (k % 4) * 1.42).toFixed(3).padStart(8);
        const y = (1.5 + Math.floor(k / 4) * 1.42).toFixed(3).padStart(8);
        const z = (-3.2 + (k % 2) * 0.5).toFixed(3).padStart(8);
        return `HETATM   ${String(33 + k).padStart(2)}  ${nm} DHZ C 295    ${x}${y}${z}  1.00 15.00           ${el}`;
      }),
      'END',
    ].join('\n');
    const cttSpec = {
      mode: 'motif',
      length: 184,
      structure: theozyme1ctt,
      unindex: 'A102,A104,A129,A132',
      fixedAtoms: { A102: 'ND1,CG,CE1', A104: 'OE1,CD', A129: 'SG,CB', A132: 'SG,CB,CA' },
      ligand: 'ZN,DHZ',
    };
    const cttBuilt = buildInput(cttSpec);
    const cttT0 = performance.now();
    const cttHandle = await engine.initialize(cttBuilt.features);
    const cttInitMs = performance.now() - cttT0;
    const cttInitPeakBytes = engine.peakMemoryUsage();
    const cttNoisy = new Float32Array(cttBuilt.features.atom_to_token_map.length * 3);
    await engine.run({ features: cttBuilt.features, X_noisy_L: cttNoisy, t: 16, n_recycle: 2, initializer: cttHandle, coordinatesOnly: true });
    const cttWarmMs = [];
    for (let i = 0; i < 3; i++) {
      const s0 = performance.now();
      await engine.run({ features: cttBuilt.features, X_noisy_L: cttNoisy, t: 8 - i, n_recycle: 2, initializer: cttHandle, coordinatesOnly: true });
      cttWarmMs.push(performance.now() - s0);
    }
    engine.releaseInitializer(cttHandle);
    const cttGenStart = performance.now();
    const cttOut = await engine.generate({ ...cttSpec, steps: 11, seed: 42, retainTrajectory: false });
    const cttGenMs = performance.now() - cttGenStart;
    const cttPeakBytes = engine.peakMemoryUsage();
    engine.trim();
    const cttAfterBytes = engine.memoryUsage();
    out.cttScale = {
      tokens: cttBuilt.features.residue_index.length,
      atoms: cttBuilt.features.atom_to_token_map.length,
      initMs: Number(cttInitMs.toFixed(1)),
      initPeakGB: Number((cttInitPeakBytes / 1e9).toFixed(3)),
      warmStepMs: cttWarmMs.map(v => Number(v.toFixed(1))),
      warmStepMeanMs: Number((cttWarmMs.reduce((a, b) => a + b, 0) / cttWarmMs.length).toFixed(1)),
      tenStepGenMs: Number(cttGenMs.toFixed(1)),
      peakGB: Number((cttPeakBytes / 1e9).toFixed(3)),
      leakedBytes: cttAfterBytes - baselineBytes,
      finite: cttOut.X_L.data.every(Number.isFinite),
      assignmentsCount: cttOut.motifAssignments?.length ?? 0,
      hasZN: cttOut.rows.some(r => r.label_comp_id === 'ZN'),
      hasDHZ: cttOut.rows.some(r => r.label_comp_id === 'DHZ'),
    };

    return out;
  } finally {
    engine.destroy();
  }
}

run().then(postMessage, error => postMessage({ error: error.stack ?? String(error) }));
