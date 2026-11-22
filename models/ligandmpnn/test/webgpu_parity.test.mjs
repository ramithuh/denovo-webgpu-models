import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { runBrowser } from "./browser-runner.mjs";
test("Real Dawn hardware: FP32 holo parity, constraints, sequence replay, FP16 rounding audit and latency", async () => {
  const r = await runBrowser("/test/gpu-harness.js");
  assert.ok(r.ok, r.error);
  console.log(JSON.stringify(r, null, 2));
  fs.mkdirSync(new URL("../.cache/", import.meta.url), { recursive: true });
  fs.writeFileSync(
    new URL("../.cache/gpu-results.json", import.meta.url),
    JSON.stringify(r, null, 2) + "\n",
  );
});
