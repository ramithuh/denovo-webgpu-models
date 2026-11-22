import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFile, readdir } from "node:fs/promises";
import { root, read } from "./helpers.mjs";
import { loadBundle, toFloat } from "../shared/loader.js";
test("row-split bundles reload with colocated scales and strict shard caps", async () => {
  execFileSync(root + ".venv/bin/python", [root + "tools/test_export.py"], {
    cwd: root,
    stdio: "pipe",
  });
  for (const precision of ["fp32", "fp16", "int8"]) {
    const b = await loadBundle({
      baseUrl: root + ".cache/multipart/weights/fixture/" + precision,
      read,
    });
    assert.deepEqual(b.weights["example.weight"].shape, [512, 256]);
    assert.equal(toFloat(b.weights["example.weight"]).length, 512 * 256);
    assert.ok(b.manifest.shards.every((s) => s.byteLength <= 131072));
  }
  const dirs = await readdir(root + "weights");
  assert.equal(dirs.length, 16);
  for (const variant of dirs)
    for (const precision of ["fp32", "fp16", "int8"]) {
      const m = JSON.parse(
        await readFile(root + `weights/${variant}/${precision}/manifest.json`),
      );
      assert.ok(m.total_params > 0);
      assert.ok(m.shards.every((s) => s.byteLength <= 25000000));
      assert.equal(
        m.total_bytes,
        m.shards.reduce((n, s) => n + s.byteLength, 0),
      );
    }
});
