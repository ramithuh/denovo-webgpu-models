#!/usr/bin/env node
/** Copy this verified runtime into a consuming application's vendor directory. */
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
const args=process.argv.slice(2),destIndex=args.indexOf('--dest');
if(destIndex<0||!args[destIndex+1])throw new Error('Usage: node sync-upstream.mjs --dest /absolute/vendor/dyna1 [--write]');
const destination=resolve(args[destIndex+1]),root=fileURLToPath(new URL('.',import.meta.url));
const files=['webgpu/index.js','webgpu/dyna1/engine.js','webgpu/dyna1/shaders.js','shared/dyna1/index.js','cpu/dyna1/index.js','LICENSE.md','README_WEBGPU.md'];
if(destination===resolve(root))throw new Error('Destination must differ from source');
for(const name of files){const source=new URL(name,import.meta.url),target=resolve(destination,name);if(args.includes('--write')){const bytes=await readFile(source);await mkdir(dirname(target),{recursive:true});await writeFile(target,bytes);}console.log(`${args.includes('--write')?'Copied':'Would copy'} ${name} -> ${target}`);}
if(args.includes('--write'))await writeFile(resolve(destination,'package.json'),JSON.stringify({name:'dyna1-webgpu-vendor',private:true,type:'module',exports:{'./webgpu':'./webgpu/index.js','./cpu':'./cpu/dyna1/index.js','./shared':'./shared/dyna1/index.js'}},null,2)+'\n');
