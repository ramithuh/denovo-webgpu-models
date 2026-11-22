#!/usr/bin/env node
/** Explicit allowlist for vendoring; dry-run unless --write is supplied. */
import {readdir,mkdir,copyFile,readFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
const root=fileURLToPath(new URL('.',import.meta.url)),args=process.argv.slice(2),write=args.includes('--write'),destArg=args.find(a=>!a.startsWith('--'));
if(!destArg)throw new Error('Usage: node sync-upstream.mjs DEST [--write]');
if(args.some(a=>a.startsWith('--')&&!['--write','--dry-run'].includes(a)))throw new Error('Unknown option');
const dest=path.resolve(destArg);if(dest===root.replace(/\/$/,'')||['shared','cpu','webgpu'].some(d=>dest.startsWith(path.join(root,d))))throw new Error('Destination overlaps runtime source');
const files=['LICENSE','README_WEBGPU.md'];
async function walk(dir){for(const e of await readdir(path.join(root,dir),{withFileTypes:true})){const rel=path.join(dir,e.name);if(e.isSymbolicLink())throw new Error(`Refusing symlink ${rel}`);if(e.isDirectory())await walk(rel);else if(e.name.endsWith('.js'))files.push(rel);else throw new Error(`Unexpected runtime asset ${rel}`);}}
for(const dir of ['shared','cpu','webgpu'])await walk(dir);
for(const file of files){console.log(`${write?'COPY':'PREVIEW'} ${file} -> ${path.join(dest,file)}`);if(write){await mkdir(path.dirname(path.join(dest,file)),{recursive:true});await copyFile(path.join(root,file),path.join(dest,file));}}
if(write){await import(pathToFileURL(path.join(dest,'webgpu/index.js')));await import(pathToFileURL(path.join(dest,'cpu/index.js')));console.log(`Validated ${files.length} vendored files`);}
