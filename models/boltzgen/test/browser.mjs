import { createServer } from "node:http";
import { createReadStream } from "node:fs";
import { access, mkdir, mkdtemp, rm, stat } from "node:fs/promises";
import { resolve, extname } from "node:path";
import { chromium } from "playwright-core";
export const root = resolve(import.meta.dirname, "..");
async function singleBrowserTask(run) {
  process.chdir(root);
  process.env.TMPDIR = ".cache/tmp";
  await mkdir(root + "/.cache/tmp", { recursive: true });
  const profile = await mkdtemp(root + "/.cache/chromium-");
  const server = createServer(async (req, res) => {
    if (req.url === "/favicon.ico") {
      res.writeHead(204);
      res.end();
      return;
    }
    try {
      let file = resolve(
        root,
        "." + decodeURIComponent(new URL(req.url, "http://localhost").pathname),
      );
      if (!file.startsWith(root + "/") || file.includes("/.git/"))
        throw Error("Forbidden");
      if ((await stat(file)).isDirectory()) file += "/index.html";
      const s = await stat(file);
      res.writeHead(200, {
        "Content-Type":
          {
            ".js": "text/javascript",
            ".mjs": "text/javascript",
            ".json": "application/json",
            ".html": "text/html",
          }[extname(file)] ?? "application/octet-stream",
        "Content-Length": s.size,
      });
      createReadStream(file).pipe(res);
    } catch {
      res.writeHead(404);
      res.end("Not found");
    }
  });
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  const url = `http://127.0.0.1:${server.address().port}`;
  let browser;
  try {
    const env = {
      ...process.env,
      TMPDIR: ".cache/tmp",
      XDG_CACHE_HOME: profile + "/cache",
      XDG_CONFIG_HOME: profile + "/config",
      NO_AT_BRIDGE: "1",
      AT_SPI_BUS_ADDRESS: "unix:path=.cache/tmp/no-accessibility-bus",
      DBUS_SESSION_BUS_ADDRESS: "unix:path=.cache/tmp/no-session-bus",
    };
    delete env.DISPLAY;
    delete env.WAYLAND_DISPLAY;
    await access("/usr/share/vulkan/icd.d/nvidia_icd.json");
    env.VK_ICD_FILENAMES = "/usr/share/vulkan/icd.d/nvidia_icd.json";
    browser = await chromium.launchPersistentContext(profile, {
      executablePath:
        process.env.CHROME_PATH ??
        "/home/ruh/.cache/ms-playwright/chromium-1243/chrome-linux-arm64/chrome",
      ignoreDefaultArgs: ["--headless", "--enable-unsafe-swiftshader"],
      headless: true,
      env,
      args: [
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
        "--disable-renderer-accessibility",
        "--enable-dawn-features=vulkan_enable_f16_on_nvidia",
        "--enable-logging=stderr",
      ],
    });
    const page = await browser.newPage();
    page.on("console", (m) => console.log("BROWSER", m.type(), m.text()));
    page.on("pageerror", (e) => console.error("BROWSER ERROR", e));
    await page.goto(url + "/test/host.html");
    // Let Chromium finish starting its native Vulkan GPU process before WebGPU.
    await page.waitForTimeout(1500);
    return await run(page, url, browser);
  } finally {
    await browser?.close();
    await new Promise((r) => server.close(r));
    await rm(profile, { recursive: true, force: true });
  }
}
export async function browserTask(run) {
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      return await singleBrowserTask(run);
    } catch (error) {
      if (
        attempt === 4 ||
        !/Hardware WebGPU adapter required|VK_ERROR_INITIALIZATION_FAILED/.test(
          error.message,
        )
      )
        throw error;
      console.warn("Retrying native Vulkan startup", attempt + 1);
    }
  }
}
