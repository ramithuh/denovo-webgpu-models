import test from "node:test";
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { readdir } from "node:fs/promises";
import { read, oracle, relRMS, root } from "./helpers.mjs";
const exec = promisify(execFile);
test("Vendor dry-run and write copy only the runtime contract and execute correctly", async () => {
  const preview = await exec(
    process.execPath,
    ["sync-upstream.mjs", ".cache/vendor-smoke"],
    { cwd: root },
  );
  assert.match(preview.stdout, /DRY RUN/);
  const written = await exec(
    process.execPath,
    ["sync-upstream.mjs", ".cache/vendor-smoke", "--write"],
    { cwd: root },
  );
  assert.match(written.stdout, /Copied/);
  assert.deepEqual(
    (await readdir(new URL(".cache/vendor-smoke/", root))).sort(),
    ["LICENSE", "README_WEBGPU.md", "cpu", "shared", "webgpu"],
  );
  const { createThermoMPNNCPU } = await import(
    "../.cache/vendor-smoke/cpu/index.js"
  );
  const { createThermoMPNNEngine } = await import(
    "../.cache/vendor-smoke/webgpu/index.js"
  );
  assert.equal(typeof createThermoMPNNEngine, "function");
  const engine = await createThermoMPNNCPU({
    baseUrl: "weights/default",
    read,
  });
  try {
    const out = await engine.predict(
        new TextDecoder().decode(await read("fixtures/small.pdb")),
      ),
      { traces } = await oracle({ id: "monomer" }, "fp32");
    assert.ok(relRMS(out.ddg, traces.ddg).relRMS < 1e-4);
  } finally {
    engine.destroy();
  }
});
