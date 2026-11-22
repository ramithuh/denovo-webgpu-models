import {cacheFixture} from './cache-fixture.mjs';
import {spawn} from 'node:child_process';
import {createServer} from 'node:http';
import {createReadStream,existsSync,statSync} from 'node:fs';
import {mkdtemp,rm} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const ROOT=fileURLToPath(new URL('../',import.meta.url));
export async function runBrowser(modulePath,{timeoutMs=900000}={}) {
 const chrome=[process.env.CHROME_PATH,`${process.env.HOME}/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome`,`${process.env.HOME}/.cache/ms-playwright/chromium-1243/chrome-linux-arm64/chrome`].find(p=>p&&existsSync(p));
 if(!chrome) throw new Error(`Chromium missing: ${chrome}`);
 let resolve;const done=new Promise(r=>resolve=r);let stderr='';
 const server=createServer((req,res)=>{
  const pathname=new URL(req.url,'http://localhost').pathname;
  const fixture=cacheFixture(pathname);if(fixture){res.end(fixture);return;}
  if(pathname==='/done'||pathname==='/log') {let body='';req.on('data',c=>body+=c);req.on('end',()=>{res.end('ok');if(pathname==='/done'){try{resolve(JSON.parse(body));}catch(e){resolve({ok:false,error:String(e)});}}else console.log('[Dawn]',body);});return;}
  if(pathname==='/') {res.setHeader('Content-Type','text/html');res.end(`<script type="module">import {run} from '${modulePath}';try {globalThis.testExecution=run();const result=await globalThis.testExecution;await fetch('/done',{method:'POST',body:JSON.stringify({ok:true,...result})});}catch(e){await fetch('/done',{method:'POST',body:JSON.stringify({ok:false,error:e.stack||String(e)})});}</script>`);return;}
  const p=path.resolve(ROOT,'.'+decodeURIComponent(pathname));
  if(!p.startsWith(ROOT)||!existsSync(p)){res.writeHead(404);res.end();return;}
  res.setHeader('Content-Type',({'.js':'text/javascript','.mjs':'text/javascript','.json':'application/json','.html':'text/html'})[path.extname(p)]??'application/octet-stream');
  res.setHeader('Content-Length',statSync(p).size);res.setHeader('Cache-Control','no-store');res.setHeader('Connection','close');
  createReadStream(p).on('error',()=>res.destroy()).pipe(res);
 });
 await new Promise(r=>server.listen(0,'127.0.0.1',r));const port=server.address().port;
 const profile=await mkdtemp(path.join(ROOT,'.cache/chrome-'));
 const env={...process.env,XDG_CONFIG_HOME:path.join(ROOT,'.cache/config'),XDG_CACHE_HOME:path.join(ROOT,'.cache/browser-cache')};delete env.DISPLAY;delete env.WAYLAND_DISPLAY;
 if(existsSync('/usr/share/vulkan/icd.d/nvidia_icd.json')) env.VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json';
 const proc=spawn(chrome,['--headless=new','--remote-debugging-port=0','--no-sandbox','--disable-gpu-sandbox','--ozone-platform=headless','--enable-unsafe-webgpu','--ignore-gpu-blocklist','--enable-features=Vulkan','--use-angle=vulkan','--use-vulkan=native','--disable-vulkan-surface','--enable-dawn-features=vulkan_enable_f16_on_nvidia',`--user-data-dir=${profile}`,`http://127.0.0.1:${port}/`],{env,stdio:['ignore','ignore','pipe']});
 proc.stderr.on('data',d=>stderr=(stderr+d).slice(-10000));proc.on('error',e=>resolve({ok:false,error:String(e)}));proc.on('exit',(code,signal)=>resolve({ok:false,error:`Chromium exited ${code} ${signal}: ${stderr}`}));
 const timer=setTimeout(()=>resolve({ok:false,error:'Timeout: '+stderr}),timeoutMs);
 try{return await done;} finally{clearTimeout(timer);if(proc.exitCode===null&&proc.signalCode===null){const closed=new Promise(r=>proc.once('close',r));proc.kill('SIGKILL');await closed;}server.closeAllConnections();await new Promise(r=>server.close(r));await rm(profile,{recursive:true,force:true});}
}
