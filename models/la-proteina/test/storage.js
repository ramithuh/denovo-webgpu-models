import {loadBundle,sha256} from '../shared/loader.js';
import {assert} from './parity.js';

export async function tinyBundle(){
  const header=new TextEncoder().encode(JSON.stringify({w:{dtype:'F32',shape:[2,2],data_offsets:[0,16]}}).padEnd(80));
  const bytes=new ArrayBuffer(8+header.length+16);new DataView(bytes).setBigUint64(0,BigInt(header.length),true);new Uint8Array(bytes,8,header.length).set(header);new Float32Array(bytes,8+header.length).set([1,2,3,4]);
  const manifest={format:'la-proteina-safetensors-v1',total_bytes:bytes.byteLength,shards:[{file:'tiny.safetensors',byteLength:bytes.byteLength,sha256:await sha256(bytes)}],tensors:{w:{shape:[2,2],dtype:'F32',parts:[{key:'w',shard:'tiny.safetensors',row_start:0,shape:[2,2]}]}}};
  return {bytes,manifest};
}
export async function storageTests(){
  assert(globalThis.caches,'Cache API must be available in hardware test browser');
  assert(navigator.storage?.getDirectory,'OPFS must be available in hardware test browser');
  const {bytes,manifest}=await tinyBundle();const result={};
  for(const cache of ['cache','opfs']){
    let reads=0;
    const baseUrl=location.origin+'/test-storage-'+cache;
    const read=async url=>{reads++;return url.endsWith('manifest.json')?new TextEncoder().encode(JSON.stringify(manifest)).buffer:bytes.slice(0);};
    const first=await loadBundle({baseUrl,read,cache});assert(first.weights.w.data[3]===4,'Cached loader first read');
    const second=await loadBundle({baseUrl,read,cache});assert(second.weights.w.data[3]===4,'Cached loader warm read');
    assert(reads===2,`${cache} did not reuse manifest/shard cache`);result[cache]={coldReads:2,warmReads:0};
  }
  // Safari Private Mode equivalent: storage permission/quota APIs throw.
  const original=Object.getOwnPropertyDescriptor(globalThis,'caches');
  Object.defineProperty(globalThis,'caches',{configurable:true,value:{open:async()=>{throw new DOMException('Denied','SecurityError');}}});
  try{
    let reads=0;const read=async url=>{reads++;return url.endsWith('manifest.json')?new TextEncoder().encode(JSON.stringify(manifest)).buffer:bytes.slice(0);};
    await loadBundle({baseUrl:location.origin+'/denied',cache:'cache',read});assert(reads===2,'Denied cache did not fall back');result.deniedCache=true;
  }finally{if(original)Object.defineProperty(globalThis,'caches',original);else delete globalThis.caches;}
  const storage=navigator.storage,own=Object.getOwnPropertyDescriptor(storage,'getDirectory');
  Object.defineProperty(storage,'getDirectory',{configurable:true,value:async()=>{throw new DOMException('Denied','SecurityError');}});
  try{
    let reads=0;const read=async url=>{reads++;return url.endsWith('manifest.json')?new TextEncoder().encode(JSON.stringify(manifest)).buffer:bytes.slice(0);};
    await loadBundle({baseUrl:location.origin+'/denied-opfs',cache:'opfs',read});assert(reads===2,'Denied OPFS did not fall back');result.deniedOPFS=true;
  }finally{if(own)Object.defineProperty(storage,'getDirectory',own);else delete storage.getDirectory;}
  return result;
}
