import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFile,readdir} from 'node:fs/promises';
const root=new URL('../',import.meta.url),target=new URL('../.cache/vendor-smoke/',import.meta.url);
test('vendor allowlist byte equality and standalone ES imports',async()=>{
  execFileSync(process.execPath,['sync-upstream.mjs','.cache/vendor-smoke','--write'],{cwd:root});
  for(const dir of ['shared','cpu','webgpu'])for(const file of await readdir(new URL(dir+'/',root)))assert.deepEqual(await readFile(new URL(dir+'/'+file,root)),await readFile(new URL(dir+'/'+file,target)));
  assert.equal(typeof(await import(new URL('webgpu/index.js',target))).createLaProteinaEngine,'function');
  assert.equal(typeof(await import(new URL('cpu/index.js',target))).createLaProteinaCPU,'function');
  assert(!(await readdir(target)).includes('weights'));
});
