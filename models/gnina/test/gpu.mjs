import {readFile,writeFile} from "node:fs/promises";
import {homedir} from "node:os";
import {serve} from "../tools/server.mjs";
const {chromium}=await import("playwright-core").catch(()=>import("../../rfdiffusion/node_modules/playwright-core/index.mjs"));
const executablePath=process.env.CHROMIUM_PATH??homedir()+"/.cache/ms-playwright/chromium-1243/chrome-linux-arm64/chrome";
const env={...process.env,VK_ICD_FILENAMES:process.env.VK_ICD_FILENAMES??"/usr/share/vulkan/icd.d/nvidia_icd.json"};
delete env.DISPLAY;delete env.WAYLAND_DISPLAY;
const server=await serve();
let browser;
try {
  browser=await chromium.launch({executablePath,headless:false,env,args:["--headless=new","--no-sandbox","--disable-gpu-sandbox","--ozone-platform=headless","--enable-unsafe-webgpu","--ignore-gpu-blocklist","--enable-features=Vulkan","--use-angle=vulkan","--use-vulkan=native","--disable-vulkan-surface"]});
  const context=await browser.newContext(),page=await context.newPage();
  page.on("console",msg=>{if(["warning","error"].includes(msg.type())) console.error(msg.text());});
  await page.goto(server.url+"/test/browser.html");
  const redocking=await page.evaluate(async ()=>(await import("./redocking.js")).redock());
  console.table(Object.entries(redocking.runs).map(([fixture,r])=>({fixture,wallMs:r.wallMs,
    top1:r.poses[0].rmsd,bestTop3:Math.min(...r.poses.slice(0,3).map(p=>p.rmsd)),cnnScore:r.poses[0].cnnScore})));
  await writeFile("fixtures/gpu_redocking.json",JSON.stringify({chromium:browser.version(),...redocking},null,2)+"\n");
  const metrics={};
  for(const precision of ["fp32","fp16","int8"]){
    console.log("Testing",precision);
    metrics[precision]=await page.evaluate(async p=>(await import("./browser.js")).suite(p),precision);
    console.log(precision,metrics[precision].warmMs);
  }
  const rec=await readFile("fixtures/184l_rec.pdb","utf8"),lig=await readFile("fixtures/184l_lig.sdf","utf8");
  for(const mode of ["cache","opfs"]){
    await page.evaluate(async mode=>{self.job=await (await import("./browser.js")).offline(mode);},mode);
    let attempts=0;
    await context.route("**/weights/**",route=>{attempts++;return route.abort();});
    const score=await page.evaluate(async ([r,l])=>self.job.run(r,l),[rec,lig]);
    if(attempts||Math.abs(score-metrics.int8.parity["184l_native"].cnnScore)>1e-6) throw Error("Worker offline cache mismatch");
    await context.unroute("**/weights/**");metrics[mode]={score,networkAttempts:attempts};
  }
  metrics.lifecycle=await page.evaluate(async ()=>(await import("./browser.js")).checkSharedDevice());
  await page.goto(server.url+"/demo/");
  await page.locator("#precision").selectOption("int8");
  await page.locator("#score").click();
  await page.locator("#results:not([hidden])").waitFor();
  if(await page.locator("#table tr").count()!==8) throw Error("Demo scores missing");
  await page.locator("#release").click();
  if(await page.locator("#status").textContent()!=="GPU released.") throw Error("Demo release failed");
  metrics.demo=true;
  await writeFile("fixtures/gpu_parity.json",JSON.stringify(metrics,null,2)+"\n");
}finally{await browser?.close();await server.close();}
