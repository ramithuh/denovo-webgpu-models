import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readdir, readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
const root = fileURLToPath(new URL("../", import.meta.url));
test("vendor --write contains only standalone runtime and docs", async () => {
  execFileSync(
    process.execPath,
    ["sync-upstream.mjs", ".cache/vendor-smoke", "--write"],
    { cwd: root },
  );
  const entries = await readdir(root + ".cache/vendor-smoke");
  assert.deepEqual(
    entries.sort(),
    ["LICENSE", "README_WEBGPU.md", "cpu", "shared", "webgpu"].sort(),
  );
  for (const dir of ["shared", "cpu", "webgpu"])
    for (const file of await readdir(root + dir)) {
      const code = await readFile(root + dir + "/" + file, "utf8");
      assert(
        !/from\s*['"](?:node:|[a-z@])|require\(|process\.|Buffer\./.test(code),
        `Runtime dependency in ${dir}/${file}`,
      );
      assert.equal(
        await readFile(
          root + ".cache/vendor-smoke/" + dir + "/" + file,
          "utf8",
        ),
        code,
      );
    }
  const cpu = await import("../.cache/vendor-smoke/cpu/index.js");
  assert.equal(typeof cpu.createEvoDiffCPU, "function");
  const gpu = await import("../.cache/vendor-smoke/webgpu/index.js");
  assert.equal(typeof gpu.createEvoDiffEngine, "function");
});
