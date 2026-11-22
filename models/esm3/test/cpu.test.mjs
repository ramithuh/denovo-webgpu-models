import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {createESM3CPU} from '../cpu/index.js';
import {checkCases} from './oracle.js';
const root=new URL('../',import.meta.url);
const read=async url=>{const b=await readFile(url);return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
test('complete ESM3 pure JS CPU reference, every block and output',async()=>{const engine=await createESM3CPU({baseUrl:new URL('weights',root).pathname,precision:'fp32',read});try{const meta=JSON.parse(await readFile(new URL('fixtures/oracle-fp32.json',root))),bin=await read(new URL('weights/oracle-fp32.bin',root));const metrics=await checkCases(engine,meta,bin);await writeFile(new URL('fixtures/cpu.json',root),JSON.stringify(metrics,null,2)+'\n');}finally{engine.destroy();}assert.equal(engine.memoryUsage().totalBytes,0);});
