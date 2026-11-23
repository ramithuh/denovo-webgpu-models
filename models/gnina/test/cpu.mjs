import {readFile,writeFile} from "node:fs/promises";
import {createGninaCPUEngine} from "../cpu/index.js";
import {verify} from "./parity.js";
const read=async path=>{const b=await readFile(path);return b.buffer.slice(b.byteOffset,b.byteOffset+b.length);};
const engine=await createGninaCPUEngine({baseUrl:"weights/default",read});
try {
  const result=await verify(engine,"fp32",p=>readFile(p,"utf8"),async p=>new Float32Array(await read(p)));
  await writeFile("fixtures/cpu_parity.json",JSON.stringify(result,null,2)+"\n");
  console.log(Object.fromEntries(Object.entries(result).map(([k,v])=>[k,[v.maxLayer,v.maxOutput]])));
} finally {engine.destroy();}
