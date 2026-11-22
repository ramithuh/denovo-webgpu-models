import test from "node:test";
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { runBrowser } from "./browser.mjs";
test(
  "Module Workers execute real ESMC; Cache API + OPFS offline and private storage fallback",
  { timeout: 7200000 },
  async () => {
    const r = await runBrowser("/test/storage-harness.js");
    assert.ok(r.ok, r.error);
    await writeFile(
      new URL("../fixtures/browser_metrics.json", import.meta.url),
      JSON.stringify(r, null, 2) + "\n",
    );
  },
);
