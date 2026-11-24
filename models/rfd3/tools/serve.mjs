#!/usr/bin/env node
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
const root=resolve(import.meta.dirname,'..'),port=Number(process.env.PORT??8080);
const types={'.html':'text/html','.js':'text/javascript','.json':'application/json','.safetensors':'application/octet-stream'};
const server=createServer(async(req,res)=>{
  try{
    let path=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
    if(path==='/')path='/demo/';
    if(path.endsWith('/'))path+='index.html';
    if(path.split('/').some(p=>p.startsWith('.')))throw new Error('Not found');
    const file=resolve(root,'.'+path);
    if(!file.startsWith(root+sep))throw new Error('Not found');
    const info=await stat(file);if(!info.isFile())throw new Error('Not found');
    res.writeHead(200,{'content-type':types[extname(file)]??'application/octet-stream','content-length':info.size});
    const stream=createReadStream(file);stream.on('error',()=>res.destroy());stream.pipe(res);
  }catch{res.writeHead(404).end('Not found');}
});
server.listen(port,'127.0.0.1',()=>console.log('http://127.0.0.1:'+port+'/demo/'));
