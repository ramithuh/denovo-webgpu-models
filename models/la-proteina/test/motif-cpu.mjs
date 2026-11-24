import { readFile, writeFile } from 'node:fs/promises';
import { createLaProteinaCPU } from '../cpu/index.js';
import { fixture, assert } from './parity.js';
import {
  ae3InputFromRef,
  checkAe3Output,
  verifyPdbMotifExtraction,
  motifSampleOptions,
  checkMotifOutput,
} from './motif-parity.js';

const root = new URL('../', import.meta.url);
const read = async (p) => {
  const b = await readFile(p instanceof URL ? p : new URL(p, root));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
const readText = (p) => readFile(new URL(p, root), 'utf8');

const manifest = JSON.parse(await readText('.cache/fixtures/motif_oracles.json'));
const extManifest = JSON.parse(await readText('.cache/fixtures/motif_extended_oracles.json'));
const pdbCache = new Map();
const getPdb = async (rel) => {
  if (!pdbCache.has(rel)) pdbCache.set(rel, await readText(`.cache/upstream/${rel}`));
  return pdbCache.get(rel);
};

// Select comprehensive FP32 CPU cases covering AE3 (2 lengths w/ layer traces) + LD4..LD7 across 3 lengths & SDE/ODE
const ae3Cases = manifest.ae3_cases.filter((c) => c.precision === 'fp32');
const ldCases = [
  ...manifest.cases.filter(
    (c) =>
      c.precision === 'fp32' &&
      ((c.task === '1ycr_L26' && ((c.steps === 1 && c.mode === 'ode') || (c.steps === 3 && c.mode === 'sde'))) ||
        (c.task === '1prw_L52' && c.steps === 1 && c.mode === 'sde')),
  ),
  ...extManifest.cases.filter((c) => c.precision === 'fp32'),
];

assert(ae3Cases.length === 2, `Expected 2 AE3 FP32 cases, got ${ae3Cases.length}`);
assert(ldCases.length === 16, `Expected 16 LD4..LD7 FP32 CPU cases, got ${ldCases.length}`);

const engine = await createLaProteinaCPU({
  baseUrl: new URL('weights', root).href,
  precision: 'fp32',
  variants: ['ae3', 'ld4', 'ld5', 'ld6', 'ld7'],
  read,
});

const ae3Results = [];
const ldResults = [];
try {
  for (const c of ae3Cases) {
    const ref = await fixture(c, read);
    const out = await engine.encodeDecode(ae3InputFromRef(c, ref), { returnIntermediates: true });
    const parity = checkAe3Output(out, ref, c);
    ae3Results.push({
      name: c.name,
      length: c.length,
      timingMs: out.timingMs,
      reconstructionRmsdAng: c.reconstruction_rmsd_ang,
      seqRecovery: c.seq_recovery,
      parity,
    });
    console.log(`[CPU AE3] ${c.name} L=${c.length} ${out.timingMs.toFixed(1)} ms PASS (coords relRMS=${parity.coords.relRMS.toExponential(2)})`);
  }

  for (const c of ldCases) {
    const ref = await fixture(c, read);
    const pdbText = await getPdb(c.pdb);
    verifyPdbMotifExtraction(pdbText, c, ref);
    const out = await engine.sample(motifSampleOptions(c, ref, pdbText));
    const parity = checkMotifOutput(out, ref, c);
    ldResults.push({
      name: c.name,
      model: c.model,
      task: c.task,
      length: c.length,
      steps: c.steps,
      mode: c.mode,
      timingMs: out.timingMs,
      motifMetrics: out.motifMetrics,
      parity,
    });
    console.log(
      `[CPU ${c.model}] ${c.name} L=${c.length} T=${c.steps} ${out.timingMs.toFixed(1)} ms PASS (coords relRMS=${parity.coords.relRMS.toExponential(2)})`,
    );
  }
} finally {
  engine.destroy();
}
assert(engine.memoryUsage().bytes === 0, 'CPU motif engine destroy leak');

await writeFile(
  new URL('.cache/fixtures/motif_cpu_metrics.json', root),
  JSON.stringify({ ae3Cases: ae3Results, cases: ldResults, skipped: 0 }, null, 2) + '\n',
);
console.log(`All ${ae3Results.length} AE3 + ${ldResults.length} LD4..LD7 CPU FP32 parity tests PASSED!`);
