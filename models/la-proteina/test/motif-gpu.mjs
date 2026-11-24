import { mkdir, writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { runBrowser } from './browser.mjs';

for (const d of ['tmp', 'browser-cache', 'browser-config']) {
  await mkdir(new URL('../.cache/' + d, import.meta.url), { recursive: true });
}
const result = await runBrowser('/test/motif-gpu-suite.js', { timeoutMs: 3600000 });
if (!result.ok) throw Error(result.error);
result.hostGPU = execFileSync('nvidia-smi', ['--query-gpu=name,driver_version', '--format=csv,noheader'], {
  encoding: 'utf8',
}).trim();
await writeFile(new URL('../.cache/fixtures/motif_gpu_metrics.json', import.meta.url), JSON.stringify(result, null, 2) + '\n');
console.log(
  JSON.stringify(
    {
      ok: result.ok,
      hostGPU: result.hostGPU,
      adapterInfo: result.adapterInfo,
      ae3Count: result.ae3Cases.length,
      ldShortCount: result.cases.length,
      ldLongCount: result.longCases.length,
      memory: result.memory,
      variantMemory: result.variantMemory,
      benchmarks: result.benchmarks,
      shaderVariants: result.shaderVariants,
      skipped: result.skipped,
    },
    null,
    2,
  ),
);
