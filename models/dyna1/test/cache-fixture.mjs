import {createHash} from 'node:crypto';
const header=JSON.stringify({w:{dtype:'I8',shape:[2,2],data_offsets:[0,4]},'w.__scale':{dtype:'F32',shape:[2],data_offsets:[4,12]}});
const n=Math.ceil(header.length/8)*8,bytes=Buffer.alloc(8+n+12,32);bytes.writeBigUInt64LE(BigInt(n));bytes.write(header,8);bytes.set([1,254,3,252],8+n);bytes.writeFloatLE(.5,12+n);bytes.writeFloatLE(2,16+n);
const manifest={format:'dyna1-safetensors-v1',config:{},total_bytes:bytes.length,shards:[{file:'test.safetensors',bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')}],tensors:{w:{key:'w',dtype:'I8',shape:[2,2],row_start:0,shard:'test.safetensors',quant:{scheme:'symmetric_per_row',axis:0,scale_tensor:'w.__scale'}}}};
export function cacheFixture(pathname){if(pathname==='/test/cache-fixture/manifest.json')return Buffer.from(JSON.stringify(manifest));if(pathname==='/test/cache-fixture/test.safetensors')return bytes;return null;}
