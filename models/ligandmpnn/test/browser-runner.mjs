import { spawn } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export async function runBrowser(modulePath, query = "") {
  const chrome = [
    process.env.CHROME_PATH,
    ...["arm64", "64"].map((a) =>
      path.join(
        os.homedir(),
        `.cache/ms-playwright/chromium-1243/chrome-linux-${a}/chrome`,
      ),
    ),
  ]
    .filter(Boolean)
    .find((p) => fs.existsSync(p));
  if (!chrome) throw Error("Chromium 1243 missing: set CHROME_PATH");
  let done, failed;
  const result = new Promise((resolve, reject) => {
    done = resolve;
    failed = reject;
  });
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, "http://localhost"),
      rel = decodeURIComponent(url.pathname);
    if (["/done", "/log"].includes(rel) && req.method === "POST") {
      let text = "";
      req.on("data", (d) => (text += d));
      req.on("end", () => {
        res.end("ok");
        if (rel === "/done") {
          try {
            done(JSON.parse(text));
          } catch (e) {
            failed(e);
          }
        } else console.log("[Dawn]", text);
      });
      return;
    }
    if (rel === "/") {
      res.setHeader("content-type", "text/html");
      res.end(`<script type="module" src="${modulePath}${query}"></script>`);
      return;
    }
    const target = path.resolve(root, "." + rel);
    if (
      !target.startsWith(root + path.sep) ||
      !fs.existsSync(target) ||
      !fs.statSync(target).isFile()
    ) {
      res.writeHead(404);
      res.end();
      return;
    }
    res.setHeader(
      "content-type",
      {
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".json": "application/json",
        ".html": "text/html",
      }[path.extname(target)] ?? "application/octet-stream",
    );
    fs.createReadStream(target).pipe(res);
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "ligandmpnn-chrome-"));
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
      `--user-data-dir=${profile}`,
      `http://127.0.0.1:${server.address().port}/`,
    ],
    { env, stdio: ["ignore", "ignore", "pipe"] },
  );
  let stderr = "";
  proc.stderr.on("data", (d) => (stderr = (stderr + d).slice(-8000)));
  proc.on("error", failed);
  proc.on("exit", (code, signal) => {
    if (code !== null) failed(Error(`Chromium exited ${code}: ${stderr}`));
  });
  const timeout = setTimeout(
    () => failed(Error("WebGPU test timed out: " + stderr)),
    300000,
  );
  try {
    const value = await result;
    if (!value.ok) value.browserStderr = stderr;
    return value;
  } finally {
    clearTimeout(timeout);
    proc.kill("SIGKILL");
    await new Promise((r) => proc.once("close", r));
    await new Promise((r) => server.close(r));
    fs.rmSync(profile, { recursive: true, force: true });
  }
}
