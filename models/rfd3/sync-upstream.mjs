#!/usr/bin/env node
import { copyFile, lstat, mkdir, mkdtemp, readdir, realpath, rename, rm, writeFile } from 'node:fs/promises';
import { dirname, join, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';

export async function syncUpstream(targetPath, { write = false } = {}) {
  const source = await realpath(import.meta.dirname), target = resolve(targetPath);
  const files = ['LICENSE', 'NOTICE', 'LICENSE-APACHE-2.0', 'README_WEBGPU.md', 'package.json', 'fixtures/architecture.json'];
  async function collect(relative) {
    for (const entry of await readdir(join(source, relative), { withFileTypes: true })) {
      const path = join(relative, entry.name);
      if (entry.isSymbolicLink()) throw new Error('Runtime source contains a symlink: ' + path);
      if (entry.isDirectory()) await collect(path);
      else if (entry.isFile()) files.push(path);
    }
  }
  for (const dir of ['shared', 'cpu', 'webgpu']) await collect(dir);
  if (!write) return { write, target, files };
  if (target === source || source.startsWith(target + sep) ||
      ['shared', 'cpu', 'webgpu', 'demo', 'tools', 'fixtures', 'weights'].some(dir =>
        target === join(source, dir) || target.startsWith(join(source, dir) + sep)))
    throw new Error('Target overlaps source or model assets');
  await mkdir(dirname(target), { recursive: true });
  if (await realpath(dirname(target)) !== dirname(target)) throw new Error('Symlinked destination parent');
  let exists = false;
  try {
    const stat = await lstat(target);
    if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error('Target must be a real directory');
    exists = true;
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  // Exclusive lock prevents two publishers racing the backup/rename sequence.
  const lock = target + '.sync-lock';
  await mkdir(lock);
  let stage, backup, published = false;
  try {
    stage = await mkdtemp(join(dirname(target), '.rfd3-stage-'));
    for (const name of files) {
      const dest = join(stage, name);
      await mkdir(dirname(dest), { recursive: true });
      await copyFile(join(source, name), dest);
    }
    await writeFile(join(stage, 'VENDOR_FILES.json'), JSON.stringify(files, null, 2) + '\n');
    if (exists) {
      backup = join(lock, 'previous');
      await rename(target, backup);
    }
    try { await rename(stage, target); published = true; }
    catch (error) { if (backup) { await rename(backup, target); backup = undefined; } throw error; }
  } finally {
    if (stage && !published) await rm(stage, { recursive: true, force: true });
    // Never remove a recovery copy if rollback itself failed.
    if (published || !backup) await rm(lock, { recursive: true, force: true });
  }
  return { write, target, files };
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const args = process.argv.slice(2), positional = args.filter(x => x !== '--write');
  if (positional.length !== 1 || positional[0].startsWith('-')) throw new Error('Usage: node sync-upstream.mjs [--write] TARGET');
  console.log(JSON.stringify(await syncUpstream(positional[0], { write: args.includes('--write') }), null, 2));
}
