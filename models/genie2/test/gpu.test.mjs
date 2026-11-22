import test from "node:test";
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { runBrowser } from "./browser.mjs";
test(
  "Real hardware WebGPU: full neural trajectories, every precision/mode/length; no skips",
  { timeout: 3600000 },
  async () => {
    for (const p of [
      ".cache/tmp",
      ".cache/browser-cache",
      ".cache/browser-config",
      ".t",
    ])
      await mkdir(new URL("../" + p, import.meta.url), { recursive: true });
    const result = await runBrowser("/test/gpu-suite.js", {
      timeoutMs: 3500000,
    });
    result.hostGPU = execFileSync(
      "nvidia-smi",
      ["--query-gpu=name,driver_version", "--format=csv,noheader"],
      { encoding: "utf8" },
    ).trim();
    await writeFile(
      new URL("../fixtures/gpu_metrics.json", import.meta.url),
      JSON.stringify(result, null, 2) + "\n",
    );
    assert.ok(result.ok, result.error);
    assert.equal(result.records.length, 18);
    assert.ok(
      result.passed,
      "Strict unmodified-PyTorch final-coordinate parity failed; see fixtures/gpu_metrics.json",
    );
  },
);
