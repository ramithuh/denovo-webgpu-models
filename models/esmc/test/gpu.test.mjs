import test from "node:test";
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { runBrowser } from "./browser.mjs";
for (const [precision, native] of [
  ["fp32", 0],
  ["fp16", 0],
  ["int8", 0],
  ["fp16", 1],
])
  test(
    `GB10 ${precision} ${native ? "shader-f16" : "core"}: full mode/trace parity and lifecycle`,
    { timeout: 7200000 },
    async () => {
      const r = await runBrowser(
        `/test/gpu-harness.js?precision=${precision}&native=${native}`,
      );
      assert.ok(r.ok, r.error);
      await writeFile(
        new URL(
          `../fixtures/gpu_${precision}${native ? "_native" : ""}_metrics.json`,
          import.meta.url,
        ),
        JSON.stringify(r, null, 2) + "\n",
      );
    },
  );
