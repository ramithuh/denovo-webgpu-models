import test from "node:test";
import assert from "node:assert/strict";
import { runBrowser } from "./browser.mjs";

test("streaming attention matches traced ESMC forwards with packed sequence IDs", async () => {
  const report = await runBrowser("/test/streaming-attention-harness.js");
  assert.ok(report.ok, report.error);
});
