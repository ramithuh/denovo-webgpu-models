import {mkdir,writeFile} from "node:fs/promises";
import {execFileSync} from "node:child_process";
import {shaders} from "../webgpu/shaders.js";
const dir=new URL("../.cache/wgsl/",import.meta.url);
await mkdir(dir,{recursive:true});
for(const [key,code] of Object.entries(shaders)){
  const file=new URL(key+".wgsl",dir);
  await writeFile(file,code);
  execFileSync("npx",["--yes","--package=naga-wasi-cli@0.1.0","naga",key+".wgsl"],{stdio:"inherit",cwd:dir});
}
