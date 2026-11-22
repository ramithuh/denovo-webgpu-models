import {createHash} from 'node:crypto';
const header=Buffer.from(JSON.stringify({'test.weight':{dtype:'F32',shape:[2,2],data_offsets:[0,16]}}).padEnd(128,' '));const bytes=Buffer.alloc(8+128+16);bytes.writeBigUInt64LE(128n);header.copy(bytes,8);for(let i=0;i<4;i++)bytes.writeFloatLE(i+1,136+i*4);
const manifest={format:'esm3-safetensors-v1',model:'cache-test',total_bytes:bytes.length,config:{},shards:[{file:'part.safetensors',bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')}],tensors:{'test.weight':{key:'test.weight',dtype:'F32',shape:[2,2],row_start:0,shard:'part.safetensors'}}};
export function cacheFixture(p){return p==='/cache-fixture/manifest.json'?JSON.stringify(manifest):p==='/cache-fixture/part.safetensors'?bytes:null;}
