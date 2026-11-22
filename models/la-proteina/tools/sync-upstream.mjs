#!/usr/bin/env node
import {cp,mkdir,stat} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('../',import.meta.url)),args=process.argv.slice(2);
const write=args.includes('--write'),arg=args.find(x=>!x.startsWith('--'));
if(args.some(x=>x.startsWith('--')&&x!=='--write'))throw Error('Usage: node sync-upstream.mjs [TARGET] [--write]');
const target=arg?path.resolve(arg):path.resolve(root,'../../../ProteinDesignSpace/web/static/vendor/la-proteina');
if(target===path.resolve(root)||['shared','cpu','webgpu'].some(p=>target.startsWith(path.join(root,p))))throw Error('Target overlaps runtime source');
const files=['shared','cpu','webgpu','package.json','LICENSE','WEIGHTS_LICENSE','THIRD_PARTY_LICENSES','NOTICE','README_WEBGPU.md'];
for(const file of files)await stat(path.join(root,file));
if(write){await mkdir(target,{recursive:true});for(const file of files)await cp(path.join(root,file),path.join(target,file),{recursive:true});}
console.log(`${write?'Copied':'Would copy'} ${files.join(', ')} to ${target}`);
