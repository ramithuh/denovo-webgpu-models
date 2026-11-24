import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";

const root = resolve(import.meta.dirname, "..");
const executable = process.env.CHROME_PATH ?? join(homedir(),
  ".cache/ms-playwright/chromium-1243/chrome-linux-arm64/chrome");
const html = `<!doctype html><script type="module">
await new Promise(r=>setTimeout(r,1500));
const worker=new Worker('/test/release-worker.js',{type:'module'});
worker.onmessage=({data})=>fetch('/result',{method:'POST',body:JSON.stringify(data)});
worker.onerror=e=>fetch('/result',{method:'POST',body:JSON.stringify({error:e.message})});
</script>`;

test("GB10 release verification: L=160 VRAM, transaction recovery, triad, non-ATP ligand+ZN, partialT, CFG", async () => {
  const profile = await mkdtemp(join(tmpdir(), "rfd3-release-"));
  const done = Promise.withResolvers();
  const requests = [];
  const server = createServer(async (request, response) => {
    requests.push(request.url);
    if (request.url === "/") {
      response.writeHead(200, { "content-type": "text/html" }).end(html);
    } else if (/^\/(?:test|webgpu|shared)\/[a-z-]+\.js$/.test(request.url)) {
      response.writeHead(200, { "content-type": "text/javascript" }).end(await readFile(join(root, request.url)));
    } else if (/^\/(?:fixtures\/(?:architecture\.json|(?:conditioning|triad|ligand|dna|rna)\.pdb)|weights\/(?:fp32|fp16|int8)\/(?:manifest\.json|[a-f0-9]{64}\.safetensors))$/.test(request.url)) {
      try { response.writeHead(200).end(await readFile(join(root, request.url))); }
      catch (error) { response.writeHead(404).end(error.message); }
    } else if (request.url === "/result" && request.method === "POST") {
      let data = "";
      for await (const chunk of request) data += chunk;
      response.writeHead(204).end();
      done.resolve(JSON.parse(data));
    } else response.writeHead(404).end();
  });
  let child, exited, timer, stderr = "";
  try {
    await new Promise((res, rej) => { server.once("error", rej); server.listen(0, "127.0.0.1", res); });
    const { port } = server.address();
    child = spawn(executable, [
      "--headless=new", "--no-sandbox", "--disable-gpu-sandbox", "--disable-dev-shm-usage",
      "--disable-background-networking", "--no-first-run", "--disable-breakpad",
      "--remote-debugging-port=0", "--no-proxy-server", "--disable-renderer-accessibility",
      "--disable-crash-reporter", "--noerrdialogs", "--ozone-platform=headless",
      "--enable-unsafe-webgpu", "--enable-webgpu-developer-features", "--ignore-gpu-blocklist", "--enable-features=Vulkan",
      "--use-angle=vulkan", "--use-vulkan=native", "--disable-vulkan-surface",
      "--user-data-dir=" + profile,
      "http://127.0.0.1:" + port,
    ], {
      cwd: root, stdio: ["ignore", "ignore", "pipe"],
      env: {
        ...process.env,
        DISPLAY: undefined, WAYLAND_DISPLAY: undefined, NO_AT_BRIDGE: "1",
        AT_SPI_BUS_ADDRESS: "unix:path=" + join(profile, "no-accessibility-bus"),
        DBUS_SESSION_BUS_ADDRESS: "unix:path=" + join(profile, "no-session-bus"),
        TMPDIR: profile, XDG_CONFIG_HOME: profile,
        XDG_CACHE_HOME: profile, CUDA_CACHE_PATH: profile, XDG_RUNTIME_DIR: profile,
        VK_ICD_FILENAMES: "/usr/share/vulkan/icd.d/nvidia_icd.json",
      },
    });
    exited = new Promise(res => child.once("close", res));
    child.stderr.on("data", data => { stderr = (stderr + data).slice(-4000); });
    child.once("error", done.reject);
    child.once("exit", code => done.reject(new Error("Chromium exited " + code + "\n" + stderr)));
    timer = setTimeout(() => done.reject(new Error("Chromium timeout after " + requests.slice(-10).join(", ") + "\n" + stderr)), 600_000);
    const result = await done.promise;
    assert.ifError(result.error && new Error(result.error + "\n" + stderr));
    console.log(JSON.stringify(result, null, 2));

    // P2-6 checks
    assert.equal(result.readUrlHasFp16, true);
    assert.ok(result.progressCount > 0 && result.progressLoaded === result.progressTotal);
    // P1-5: L=160 peak VRAM must drop dramatically below 2.5 GB (was 7.02 GB!) with 0 leaked bytes
    assert.ok(result.l160.full160PeakGB < 2.5, `Expected L=160 peak < 2.5 GB, got ${result.l160.full160PeakGB} GB`);
    assert.equal(result.l160.leakedBytesAfter160, 0);
    // P1-5: transaction / RangeError recovery with 0 leaked transient bytes + P1-4 low-step L=16 seed=7
    assert.equal(result.recovery.oversizedErr?.name, "RangeError");
    assert.equal(result.recovery.txErr?.name, "RangeError");
    assert.equal(result.recovery.leakedAfterOversized, 0);
    assert.equal(result.recovery.leakedMidTransaction, 0);
    assert.equal(result.recovery.leakedAfterRelease, 0);
    assert.equal(result.recovery.leakedAfterRecovery, 0);
    assert.equal(result.recovery.recoveredFinite, true);
    assert.equal(result.recovery.p1_4_lowStepL16Seed7Passed, true);
    // P1-1: triad unindexed catalytic motif
    assert.equal(result.triad.finite, true);
    assert.equal(result.triad.assignments.length, 3);
    assert.equal(result.triad.hasHIS, true);
    assert.equal(result.triad.hasSER, true);
    assert.equal(result.triad.hasASP, true);
    // P1-2: non-ATP ligand + ZN
    assert.equal(result.ligandZn.finite, true);
    assert.equal(result.ligandZn.hasLig, true);
    assert.equal(result.ligandZn.hasZn, true);
    assert.equal(result.ligandZn.originFinite, true);
    // P1-3: partialT
    assert.equal(result.partial.finite, true);
    assert.ok(result.partial.originX < 5, `partial origin should be centered on A1-2, got ${result.partial.originX}`);
    assert.deepEqual(result.partial.refPlddt, [1, 1, 0]);
    // P2-3: CFG + P3-2 entropy B-factor
    assert.equal(result.cfg.finite, true);
    assert.equal(result.cfg.bFactorsFinite, true);
  } finally {
    clearTimeout(timer);
    if (child) {
      child.kill("SIGTERM");
      const kill = setTimeout(() => child.kill("SIGKILL"), 2000);
      await exited;
      clearTimeout(kill);
    }
    server.closeAllConnections();
    await new Promise(res => server.close(res));
    await rm(profile, { recursive: true, force: true });
  }
});
