import {createServer} from 'node:http';
import {createReadStream} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('../',import.meta.url));
const server=createServer((req,res)=>{
  let pathname;try{pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname);}catch{res.writeHead(400);res.end();return;}
  if(pathname.endsWith('/'))pathname+='index.html';
  const file=path.resolve(root,'.'+pathname);
  if(!file.startsWith(root)||pathname.includes('/.')){res.writeHead(404);res.end();return;}
  res.setHeader('Content-Type',({'.js':'text/javascript','.json':'application/json','.html':'text/html','.css':'text/css'})[path.extname(file)]??'application/octet-stream');
  createReadStream(file).on('error',()=>{res.writeHead(404);res.end();}).pipe(res);
});
const port=Number(process.env.PORT??8080);server.listen(port,'127.0.0.1',()=>console.log(`La-Proteina demo: http://127.0.0.1:${port}/demo/`));
