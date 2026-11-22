import test from "node:test";
import assert from "node:assert/strict";
import { runBrowser } from "./browser-runner.mjs";
test("Dawn compiles FP32, packed-half, shader-f16, GELU and normalization WGSL", async () => {
  const r = await runBrowser("/test/gpu-harness.js", "?shaders");
  assert.ok(r.ok, r.error);
  console.log(r);
});
