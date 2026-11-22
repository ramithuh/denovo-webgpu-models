import {cp,mkdir,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('../',import.meta.url));
export async function vendorSync(destination){const dest=path.resolve(destination);if(dest===root)throw new Error('Destination cannot be the source package');await mkdir(dest,{recursive:true});for(const name of ['shared','cpu','webgpu','LICENSE.md','README_WEBGPU.md'])await cp(path.join(root,name),path.join(dest,name),{recursive:true});await writeFile(path.join(dest,'package.json'),JSON.stringify({name:'esm3-webgpu-vendor',type:'module',private:true}));return dest;}
if(process.argv[1]===fileURLToPath(import.meta.url)){if(!process.argv[2])throw new Error('Pass destination directory');console.log(await vendorSync(process.argv[2]));}
