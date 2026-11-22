import { chromium } from "playwright-core";
import { mkdir, writeFile, mkdtemp, rm, readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { serve } from "../tools/server.mjs";
const root = fileURLToPath(new URL("../", import.meta.url));
for (const d of ["tmp", "chromium-profile", "xdg-config", "xdg-cache"])
  await mkdir(root + ".cache/" + d, { recursive: true });
await mkdir(root + ".t", { recursive: true });
process.env.TMPDIR = root + ".t";
const browserEnv = {
  ...process.env,
  TMPDIR: root + ".t",
  XDG_CONFIG_HOME: root + ".cache/xdg-config",
  XDG_CACHE_HOME: root + ".cache/xdg-cache",
  VK_ICD_FILENAMES: "/usr/share/vulkan/icd.d/nvidia_icd.json",
};
delete browserEnv.DISPLAY;
delete browserEnv.WAYLAND_DISPLAY;
const profile = await mkdtemp(root + ".cache/browser-");
const host = await serve();
let context;
try {
  context = await chromium.launchPersistentContext(profile, {
    executablePath:
      process.env.CHROME_PATH ??
      "/home/ruh/.cache/ms-playwright/chromium-1243/chrome-linux-arm64/chrome",
    headless: true,
    acceptDownloads: true,
    ignoreDefaultArgs: true,
    env: browserEnv,
    args: [
      "--remote-debugging-pipe",
      "--user-data-dir=" + profile,
      "--enable-automation",
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-background-networking",
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
      "about:blank",
    ],
  });
  const page = await context.newPage();
  page.on("console", (m) => console.log(m.type(), m.text()));
  page.on("pageerror", (e) => console.error(e));
  await page.goto(host.url + "/test/browser.html");
  const metrics = await page.evaluate(async () => {
    const { run } = await import("./browser-suite.js");
    return run();
  });
  metrics.browserVersion = context.browser().version();
  const session = await context.browser().newBrowserCDPSession();
  metrics.hardware = (await session.send("SystemInfo.getInfo")).gpu.devices;
  await session.detach();
  metrics.workerOffline = await workerOffline(page, context);
  metrics.demo = await demoTest(page, host.url);
  await writeFile(
    root + "fixtures/gpu_metrics.json",
    JSON.stringify(metrics, null, 2) + "\n",
  );
  console.log("Hardware WebGPU tests passed; no skips.");
} finally {
  await context?.close();
  await host.close();
  await rm(profile, { recursive: true, force: true });
}

async function workerOffline(page, context) {
  await page.evaluate(async () => {
    const { makeWorker } = await import("./worker-client.js");
    window.testWorker = await makeWorker();
    window.testInput = await (await fetch("../fixtures/small.pdb")).text();
    const meta = await (
        await fetch("../fixtures/oracle_monomer_fp32.json")
      ).json(),
      binary = await (
        await fetch("../fixtures/oracle_monomer_fp32.bin")
      ).arrayBuffer(),
      t = meta.tensors.ddg;
    window.testExpected = new Float32Array(
      binary,
      t.offset,
      t.byteLength / 4,
    ).slice();
  });
  const results = {};
  try {
    for (const cache of ["cache", "opfs"]) {
      await page.evaluate(async (cache) => {
        await window.testWorker.call("init", {
          options: {
            baseUrl: location.origin + "/weights/default",
            precision: "fp32",
            cache,
          },
        });
        await window.testWorker.call("predict", { text: window.testInput });
        await window.testWorker.call("destroy");
      }, cache);
      await context.setOffline(true);
      try {
        results[cache] = await page.evaluate(async (cache) => {
          let blocked = false;
          try {
            await fetch("../fixtures/small.pdb", { cache: "reload" });
          } catch {
            blocked = true;
          }
          if (!blocked) throw new Error("Network was not blocked");
          await window.testWorker.call("init", {
            options: {
              baseUrl: location.origin + "/weights/default",
              precision: "fp32",
              cache,
            },
          });
          const result = await window.testWorker.call("predict", {
            text: window.testInput,
            options: {
              mode: "targeted",
              mutations: [{ chain: "A", number: 96, mutation: "A" }],
            },
          });
          if (
            !(result.ddg instanceof Float32Array) ||
            result.ddg.length !== 240 ||
            result.memory.transientBuffers !== 0
          )
            throw new Error("Worker output or buffer cleanup failed");
          let error = 0,
            ref = 0;
          for (let i = 0; i < result.ddg.length; i++) {
            error += (result.ddg[i] - window.testExpected[i]) ** 2;
            ref += window.testExpected[i] ** 2;
          }
          const relRMS = Math.sqrt(error / ref);
          if (!(relRMS < 1e-4)) throw new Error("Worker parity failure");
          if (result.scores[0] !== result.ddg[0])
            throw new Error("Worker targeted score mismatch");
          const destroyed = await window.testWorker.call("destroy");
          if (destroyed.memory.totalBytes !== 0)
            throw new Error("Worker destroy leak");
          return {
            networkBlocked: blocked,
            relRMS,
            transferableTypedArrays: true,
            zeroTransientBuffers: true,
            destroyedBytes: 0,
          };
        }, cache);
      } finally {
        await context.setOffline(false);
      }
    }
  } finally {
    await page.evaluate(() => {
      window.testWorker.close();
    });
  }
  return results;
}
async function demoTest(page, url) {
  await page.goto(url + "/demo/");
  if (
    (await page
      .locator(".layout")
      .evaluate((el) => getComputedStyle(el).display)) !== "grid"
  )
    throw new Error("Demo stylesheet did not load");
  await page.click("#example");
  await page.waitForFunction(() =>
    document.querySelector("#structure").value.includes("ATOM"),
  );
  await page.click("#run");
  await page.waitForFunction(
    () => document.querySelector("#status").textContent.startsWith("Done."),
    {},
    { timeout: 120000 },
  );
  const summary = await page.locator("#summary").textContent();
  if (!summary.includes("12 residues"))
    throw new Error("Demo saturation failed");
  await page.screenshot({ path: root + ".cache/demo.png", fullPage: true });
  for (let attempt = 0; attempt < 3; attempt++) {
    const download = page.waitForEvent("download");
    await page.click("#csv");
    const file = await download;
    await file.saveAs(root + ".cache/demo.csv");
    const contents = await readFile(root + ".cache/demo.csv", "utf8");
    if (contents.trim().split("\n").length !== 241)
      throw new Error("Demo CSV contents incomplete");
  }
  await page.selectOption("#mode", "targeted");
  await page.fill("#mutations", "A:96:A");
  await page.selectOption("#precision", "fp16");
  await page.click("#run");
  await page.waitForFunction(
    () => document.querySelector("#status").textContent.startsWith("Done."),
    {},
    { timeout: 120000 },
  );
  if ((await page.locator("#rows tr").count()) !== 1)
    throw new Error("Demo targeted rows failed");
  await page.click("#release");
  await page.waitForFunction(
    () =>
      document.querySelector("#status").textContent === "GPU buffers released.",
  );
  return {
    ssm: true,
    targeted: true,
    precisionSelector: true,
    csvDownload: true,
    releaseGPU: true,
  };
}
