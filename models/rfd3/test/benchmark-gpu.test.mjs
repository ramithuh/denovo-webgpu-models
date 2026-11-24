import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { createServer } from "node:http";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join, relative, resolve } from "node:path";
import test from "node:test";

const root = resolve(import.meta.dirname, "..");
const executable = process.env.CHROME_PATH ?? join(homedir(),
  ".cache/ms-playwright/chromium-1243/chrome-linux-arm64/chrome");
const modules = new Set(["/test/browser-worker.js", "/shared/shard.js", "/webgpu/resources.js",
  "/demo/worker.js", "/demo/gpu-worker.js", "/shared/structure.js"]);
const fixture = Buffer.from("browser storage integrity test\n");
const html = `<!doctype html><script type="module">
await new Promise(r=>setTimeout(r,1500));
const worker=new Worker('/test/benchmark-worker.js',{type:'module'});
worker.onmessage=({data})=>fetch('/result',{method:'POST',body:JSON.stringify(data)});
worker.onerror=e=>fetch('/result',{method:'POST',body:JSON.stringify({error:e.message})});
</script>`;

test("GB10 full trajectory benchmark", async () => {
  const profile = await mkdtemp(join(tmpdir(), "rfd3-hardware-"));
  const done = Promise.withResolvers();
  let networkReads = 0;
  const requests = [];
  const server = createServer(async (request, response) => {
    requests.push(request.url);
    if (request.url === "/") {
      response.writeHead(200, { "content-type": "text/html" }).end(html);
    } else if ((/^\/(?:test|webgpu|shared)\/[a-z-]+\.js$/.test(request.url)) || /^\/(?:test\/linear-worker|webgpu\/(?:linear|weights)|shared\/(?:weights|metrics))\.js$/.test(request.url)) {
      response.writeHead(200, { "content-type": "text/javascript" }).end(
        await readFile(join(root, request.url)));
    } else if (/^\/(?:fixtures\/(?:architecture\.json|(?:conditioning|ligand|dna|rna)\.pdb|oracle\/[a-z0-9-]+\.json)|weights\/(?:fp32|fp16|int8)\/(?:manifest\.json|[a-f0-9]{64}\.safetensors))$/.test(request.url)) {
      try { response.writeHead(200).end(await readFile(join(root, request.url))); }
      catch (error) { response.writeHead(404).end(error.message); }
    } else if (request.url === "/fixture") {
      networkReads++;
      response.writeHead(200, { "cache-control": "no-store" }).end(fixture);
    } else if (request.url === "/partial") {
      response.writeHead(200, { "cache-control": "no-store" });
      response.write(fixture.subarray(0, 1));
    } else if (request.url === "/result" && request.method === "POST") {
      let data = "";
      for await (const chunk of request) data += chunk;
      response.writeHead(204).end();
      done.resolve(JSON.parse(data));
    } else response.writeHead(404).end();
  });
  let child;
  let exited;
  let timer;
  let stderr = "";
  try {
    await new Promise((resolve, reject) => {
      server.once("error", reject);
      server.listen(0, "127.0.0.1", resolve);
    });
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
    exited = new Promise(resolve => child.once("close", resolve));
    child.stderr.on("data", data => { stderr = (stderr + data).slice(-4000); });
    child.once("error", done.reject);
    child.once("exit", code => done.reject(new Error("Chromium exited " + code + "\n" + stderr)));
    timer = setTimeout(() => done.reject(new Error("Chromium timeout after " + requests.join(", ") + "\n" + stderr)), 1_800_000);
    const result = await done.promise;
    assert.ifError(result.error && new Error(result.error + "\n" + stderr));
    await writeFile(join(root,'fixtures/gpu-generation.json'),JSON.stringify(result,null,2)+'\n');
    assert.ok(result.results.every(r=>r.finite));
    const analysis=spawnSync('/home/ruh/miniconda3/envs/foundry/bin/python',['-B','tools/analyze_geometry.py'],{cwd:root,encoding:'utf8',timeout:120000});assert.equal(analysis.status,0,analysis.stderr);
    const geometry=JSON.parse(await readFile(join(root,'fixtures/generation-geometry.json')));
    assert.equal(geometry.length,result.results.length);
    for(const row of geometry){
      const label=row.precision+' '+row.case;
      assert.equal(row.chainBreaks,0,label+' chain breaks');
      assert.equal(row.severeClashes,0,label+' heavy-atom overlaps below 1.5 Å');
      assert.equal(row.biopythonAtoms,row.outputAtoms,label+' atom preservation');
      if(row.caMean!=null)assert.ok(row.caMean>3.6&&row.caMean<4.0,label+' mean Cα distance');
      if(row.motifRmsd!=null)assert.ok(row.motifRmsd<1e-4,label+' held motif RMSD');
      assert.ok(row.secondaryStructurePSEA,label+' secondary structure annotation');
      if(row.cyclicClosure)assert.ok(row.cyclicClosure.CN>1.2&&row.cyclicClosure.CN<1.5,label+' peptide closure');
    }
    console.log(JSON.stringify(result.results.map(({pdb,mmcif,...rest})=>rest)));
  } finally {
    clearTimeout(timer);
    if (child) {
      child.kill("SIGTERM");
      const kill = setTimeout(() => child.kill("SIGKILL"), 2000);
      await exited;
      clearTimeout(kill);
    }
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    await rm(profile, { recursive: true, force: true });
  }
});
