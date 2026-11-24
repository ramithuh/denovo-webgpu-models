import { createServer } from "node:http";
import { readFile, mkdir, access, mkdtemp, rm } from "node:fs/promises";
import { resolve, extname } from "node:path";
import { chromium } from "playwright-core";
import { root } from "./helpers.mjs";
async function singleBrowserTask(run) {
  const chrome =
    process.env.CHROME_PATH ??
    "/home/ruh/.cache/ms-playwright/chromium-1243/chrome-linux-arm64/chrome";
  await access(chrome);
  await mkdir(root + ".cache/chromium", { recursive: true });
  await mkdir(root + ".cache/tmp", { recursive: true });
  process.chdir(root);
  process.env.TMPDIR = ".cache/tmp";
  const server = createServer(async (req, res) => {
    if (req.url === "/favicon.ico") {
      res.writeHead(204);
      res.end();
      return;
    }
    try {
      const file = resolve(
        root,
        "." + decodeURIComponent(new URL(req.url, "http://localhost").pathname),
      );
      if (!file.startsWith(root) || file.includes("/.git/"))
        throw new Error("Forbidden path");
      const data = await readFile(
        file.endsWith("/") ? file + "index.html" : file,
      );
      res.writeHead(200, {
        "Content-Type":
          {
            ".js": "text/javascript",
            ".mjs": "text/javascript",
            ".html": "text/html",
            ".json": "application/json",
          }[extname(file)] ?? "application/octet-stream",
      });
      res.end(data);
    } catch {
      res.writeHead(404);
      res.end("Not found");
    }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const url = `http://127.0.0.1:${server.address().port}`;
  const profile = await mkdtemp("/dev/shm/rfd-browser-profile-");
  await mkdir(profile + "/tmp", { recursive: true });
  let browser, timer;
  try {
    const env = {
      ...process.env,
      TMPDIR: profile + "/tmp",
      XDG_CACHE_HOME: profile + "/cache",
      XDG_CONFIG_HOME: profile + "/config",
      // Isolate headless Chrome from the host desktop/accessibility session.
      NO_AT_BRIDGE: "1",
      DBUS_SESSION_BUS_ADDRESS: `unix:path=${profile}/tmp/no-session-bus`,
      AT_SPI_BUS_ADDRESS: `unix:path=${profile}/tmp/no-accessibility-bus`,
    };
    delete env.DISPLAY;
    delete env.WAYLAND_DISPLAY;
    try {
      await access("/usr/share/vulkan/icd.d/nvidia_icd.json");
      env.VK_ICD_FILENAMES = "/usr/share/vulkan/icd.d/nvidia_icd.json";
    } catch {}
    browser = await chromium.launchPersistentContext(profile, {
      executablePath: chrome,
      ignoreDefaultArgs: ["--headless", "--enable-unsafe-swiftshader"],
      headless: true,
      env,
      args: [
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu-sandbox",
        "--disable-renderer-accessibility",
        "--password-store=basic",
        "--use-mock-keychain",
        "--ozone-platform=headless",
        "--enable-unsafe-webgpu",
        "--ignore-gpu-blocklist",
        "--enable-features=Vulkan",
        "--use-angle=vulkan",
        "--use-vulkan=native",
        "--disable-vulkan-surface",
        "--enable-dawn-features=vulkan_enable_f16_on_nvidia",
        "--enable-logging=stderr",
      ],
    });
    const page = await browser.newPage();
    page.on("console", (message) => {
      if (message.type() === "error") console.error(message.text());
    });
    page.on("pageerror", (error) => console.error(error));
    await page.goto(url + "/test/host.html");
    return await Promise.race([
      run(page, url, browser),
      new Promise((_, reject) => {
        timer = setTimeout(
          () => reject(Error("Hardware browser task exceeded 180 seconds")),
          180000,
        );
      }),
    ]);
  } finally {
    clearTimeout(timer);
    await browser?.close();
    await new Promise((resolve) => server.close(resolve));
    await rm(profile, { recursive: true, force: true });
  }
}

export async function browserTask(run) {
  // A native Vulkan context can fail during Chromium startup on this driver.
  // A fresh process may recover; numerical, shader, fixture and software-adapter
  // failures are never retried or converted into a passing result.
  for (let attempt = 1; attempt <= 5; attempt++) {
    try {
      return await singleBrowserTask(run);
    } catch (error) {
      if (
        attempt === 5 ||
        !/external Instance reference|Hardware WebGPU adapter required|No adapter|VK_ERROR_INITIALIZATION_FAILED|vkCreateDevice failed/.test(
          error.message,
        )
      )
        throw error;
      console.warn(
        `Native Vulkan startup retry ${attempt}/5: ${error.message.split("\n")[0]}`,
      );
    }
  }
}
