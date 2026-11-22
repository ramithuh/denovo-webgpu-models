import test from "node:test";
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { readdir, readFile, writeFile } from "node:fs/promises";
import { runBrowser } from "./browser.mjs";
const exec = promisify(execFile),
  root = new URL("../", import.meta.url);
test(
  "Vendor allowlist dry-run and local write smoke; demo runs every mode",
  { timeout: 7200000 },
  async () => {
    const dry = await exec(
      process.execPath,
      ["sync-upstream.mjs", ".cache/vendor-smoke"],
      { cwd: root },
    );
    assert.match(dry.stdout, /WOULD COPY/);
    assert.doesNotMatch(
      dry.stdout,
      /^WOULD COPY (weights|fixtures|\.venv|tools)\//m,
    );
    await exec(
      process.execPath,
      ["sync-upstream.mjs", ".cache/vendor-smoke", "--write"],
      { cwd: root },
    );
    assert.deepEqual(
      (await readdir(new URL(".cache/vendor-smoke/", root))).sort(),
      ["LICENSE", "README_WEBGPU.md", "cpu", "shared", "webgpu"],
    );
    const r = await runBrowser("/test/demo-harness.js");
    assert.ok(r.ok, r.error);
    await writeFile(
      new URL("fixtures/demo_metrics.json", root),
      JSON.stringify(r, null, 2) + "\n",
    );
  },
);
