import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import os from "node:os";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
const root = fileURLToPath(new URL("../", import.meta.url));
export async function browserRun(script, { timeout = 240000 } = {}) {
  const chrome =
    process.env.CHROME_PATH ??
    [
      `${process.env.HOME}/.cache/ms-playwright/chromium-1243/chrome-linux-arm64/chrome`,
      `${process.env.HOME}/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome`,
    ].find(fs.existsSync);
  if (!chrome)
    throw Error(
      "Chromium missing; set CHROME_PATH. Hardware tests are not silently skipped.",
    );
  let resolve;
  const done = new Promise((r) => (resolve = r));
  const server = http.createServer((req, res) => {
    if (req.method === "POST" && ["/done", "/log"].includes(req.url)) {
      let body = "";
      req.on("data", (d) => (body += d));
      req.on("end", () => {
        res.end("ok");
        if (req.url === "/log") console.log("[Dawn]", body);
        else {
          try {
            resolve(JSON.parse(body));
          } catch {
            resolve({ error: body });
          }
        }
      });
      return;
    }
    if (req.url === "/") {
      res.setHeader("Content-Type", "text/html");
      res.end(
        `<!doctype html><script type="module">const log=msg=>fetch('/log',{method:'POST',body:String(msg)});try{const result=await(async()=>{${script}})();await fetch('/done',{method:'POST',body:JSON.stringify({ok:true,...result})});}catch(e){await fetch('/done',{method:'POST',body:JSON.stringify({ok:false,error:e.stack??String(e)})});}</script>`,
      );
      return;
    }
    const file = path.resolve(
      root,
      "." + decodeURIComponent(req.url.split("?")[0]),
    );
    if (
      !file.startsWith(root) ||
      !fs.existsSync(file) ||
      !fs.statSync(file).isFile()
    ) {
      res.statusCode = 404;
      res.end();
      return;
    }
    res.setHeader(
      "Content-Type",
      {
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".json": "application/json",
      }[path.extname(file)] ?? "application/octet-stream",
    );
    fs.createReadStream(file).pipe(res);
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const port = server.address().port,
    profile = fs.mkdtempSync(path.join(os.tmpdir(), "lasermpnn-chrome-"));
  const env = { ...process.env };
  delete env.DISPLAY;
  delete env.WAYLAND_DISPLAY;
  if (fs.existsSync("/usr/share/vulkan/icd.d/nvidia_icd.json"))
    env.VK_ICD_FILENAMES = "/usr/share/vulkan/icd.d/nvidia_icd.json";
  const proc = spawn(
    chrome,
    [
      "--headless=new",
      "--no-sandbox",
      "--disable-gpu-sandbox",
      "--ozone-platform=headless",
      "--enable-unsafe-webgpu",
      "--ignore-gpu-blocklist",
      "--enable-features=Vulkan",
      "--use-angle=vulkan",
      "--use-vulkan=native",
      "--disable-vulkan-surface",
      "--enable-dawn-features=vulkan_enable_f16_on_nvidia",
      ...(process.env.CHROME_EXTRA_FLAGS?.split(" ") ?? []),
      `--user-data-dir=${profile}`,
      `http://127.0.0.1:${port}/`,
    ],
    { env, stdio: ["ignore", "ignore", "pipe"] },
  );
  let stderr = "";
  proc.stderr.on("data", (d) => (stderr = (stderr + d).slice(-6000)));
  proc.on("error", (e) => resolve({ ok: false, error: String(e) }));
  proc.on("exit", (code) =>
    resolve({ ok: false, error: `Chromium exit ${code}: ${stderr}` }),
  );
  const timer = setTimeout(
    () => resolve({ ok: false, error: `Browser timed out: ${stderr}` }),
    timeout,
  );
  try {
    const result = await done;
    if (process.env.DEBUG_CHROME) result.stderr = stderr;
    return result;
  } finally {
    clearTimeout(timer);
    proc.kill("SIGKILL");
    if (proc.exitCode === null && proc.signalCode === null)
      await new Promise((r) => proc.once("close", r));
    await new Promise((r) => server.close(r));
    fs.rmSync(profile, { recursive: true, force: true });
  }
}
export const gpuSetup = `const adapter=await navigator.gpu?.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw Error('No WebGPU adapter');const info=adapter.info;if(info.isFallbackAdapter||/swiftshader|llvmpipe|lavapipe/i.test(info.description))throw Error('Hardware GPU required');const device=await adapter.requestDevice({requiredFeatures:adapter.features.has('shader-f16')?['shader-f16']:[]});await log(JSON.stringify({vendor:info.vendor,architecture:info.architecture,device:info.device,description:info.description}));`;
