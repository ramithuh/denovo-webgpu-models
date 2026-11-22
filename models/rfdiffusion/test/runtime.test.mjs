import test from "node:test";
import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { root } from "./helpers.mjs";
import { parsePDB, parseContig, writePDB } from "../shared/structure.js";
import { requestHardwareAdapter } from "../webgpu/device.js";
import { GPUAlgebra } from "../webgpu/algebra.js";
const exec = promisify(execFile);
test("runtime import graph has no packages or Node built-ins and vendor sync is bounded", async () => {
  for (const dir of ["shared", "cpu", "webgpu"])
    for (const file of await readdir(root + dir)) {
      if (!/\.(mjs|js)$/.test(file)) continue;
      const source = await readFile(root + dir + "/" + file, "utf8");
      for (const match of source.matchAll(
        /(?:from\s*|import\s*\(?\s*)['"]([^'"]+)['"]/g,
      ))
        assert.ok(match[1].startsWith("."), `${dir}/${file}: ${match[1]}`);
    }
  const { stdout } = await exec(
    process.execPath,
    [root + "sync-upstream.mjs", "--write", root + ".cache/vendor-smoke"],
    { cwd: root },
  );
  const copied = JSON.parse(stdout);
  assert.ok(copied.write);
  assert.ok(copied.files.includes("webgpu/index.js"));
  for (const name of copied.files) {
    assert.match(name, /^(shared\/|cpu\/|webgpu\/|LICENSE$|README_WEBGPU.md$)/);
    assert.deepEqual(
      await readFile(root + name),
      await readFile(root + ".cache/vendor-smoke/" + name),
    );
  }
});
test("PDB and motif contigs preserve mapped backbone coordinates", () => {
  const xyz = Float32Array.from({ length: 4 * 12 }, (_, i) => i * 0.1),
    seq = [0, 1, 2, 3],
    pdb = parsePDB(writePDB(xyz, seq)),
    mapping = parseContig("2/A2-3/0 3", { pdb });
  assert.equal(mapping.length, 7);
  assert.equal(mapping[2].source.number, 2);
  assert.equal(mapping[3].source.number, 3);
  assert.equal(mapping[4].chain, 1);
  assert.ok(mapping[4].idx >= 200);
  assert.throws(() => parseContig("A99-100", { pdb }), /Missing backbone/);
});
test("software adapters and oversized GPU allocations fail explicitly", async () => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  try {
    Object.defineProperty(globalThis, "navigator", {
      configurable: true,
      value: {
        gpu: {
          requestAdapter: async () => ({
            info: { vendor: "swiftshader", isFallbackAdapter: true },
          }),
        },
      },
    });
    await assert.rejects(requestHardwareAdapter(), /Software WebGPU/);
  } finally {
    Object.defineProperty(globalThis, "navigator", descriptor);
  }
  const ops = new GPUAlgebra({
    limits: { maxBufferSize: 4, maxStorageBufferBindingSize: 4 },
    addEventListener() {},
    removeEventListener() {},
  });
  assert.throws(() => ops.alloc([2]), /exceeds/);
  assert.deepEqual(ops.memoryUsage(), { bytes: 0, transientBuffers: 0 });
  ops.destroy();
});
