import test from "node:test";
import assert from "node:assert/strict";
import { readFile, readdir, mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { createHash } from "node:crypto";
import { parseSafetensors, createReader } from "../shared/loader.js";
import { createFeatures, parsePDB } from "../shared/input.js";
import { randomInputs, sigmaSchedule } from "../shared/random.js";
import { vendorSync } from "../tools/sync-upstream.mjs";
const root = resolve(import.meta.dirname, "..");
async function walk(dir) {
  const out = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = resolve(dir, entry.name);
    out.push(...(entry.isDirectory() ? await walk(path) : [path]));
  }
  return out;
}

test("runtime is zero-dependency ESM and WGSL uses no dynamically indexed const arrays", async () => {
  for (const folder of ["cpu", "shared", "webgpu"])
    for (const path of await walk(resolve(root, folder))) {
      const source = await readFile(path, "utf8");
      assert.doesNotMatch(source, /from\s*['"](?:node:|[^.\/])/);
      assert.doesNotMatch(source, /require\s*\(/);
      assert.doesNotMatch(source, /const\s+\w+\s*:\s*array</);
    }
});

test("all six weight bundles have bounded shards, verified SHA-256, and colocated INT8 scales", async () => {
  for (const variant of ["diverse", "adherence"])
    for (const precision of ["fp32", "fp16", "int8"]) {
      const base = resolve(root, "weights", variant, precision),
        manifest = JSON.parse(await readFile(base + "/manifest.json", "utf8"));
      let total = 0,
        tensors = 0;
      assert.equal(manifest.num_tensors, 4739);
      assert.equal(
        manifest.upstream_commit,
        "6472b44e8072187e6eaa51271ae6838245206013",
      );
      for (const shard of manifest.shards) {
        const bytes = await readFile(base + "/" + shard.file);
        assert.equal(bytes.length, shard.byteLength);
        assert.ok(bytes.length < 25_000_000);
        assert.equal(
          createHash("sha256").update(bytes).digest("hex"),
          shard.sha256,
        );
        const ts = parseSafetensors(
          bytes.buffer.slice(
            bytes.byteOffset,
            bytes.byteOffset + bytes.byteLength,
          ),
        );
        for (const [name, t] of Object.entries(ts)) {
          if (!name.endsWith(".__scale")) tensors++;
          if (t.dtype === "I8")
            assert.equal(ts[name + ".__scale"]?.dtype, "F32");
        }
        total += bytes.length;
      }
      assert.equal(total, manifest.total_bytes);
      assert.equal(tensors, manifest.num_tensors);
      assert.equal(
        manifest.programs,
        undefined,
        "Weights must not depend on a captured program",
      );
    }
});

test("dynamic features, target identities, input validation, and random schedules", async () => {
  for (const L of [1, 8, 16, 24, 32, 64]) {
    const f = createFeatures({ L });
    assert.deepEqual(f.get("f:token_pad_mask").shape, [1, L]);
    assert.equal(
      f.get("f:atom_pad_mask").data.reduce((a, b) => a + b),
      14 * L,
    );
    assert.equal(f.get("f:atom_to_token").shape[1] % 32, 0);
    assert.equal(f.metadata.atoms.length, L * 14);
  }
  const pdb = await readFile(root + "/fixtures/1crn-12.pdb", "utf8");
  assert.equal(parsePDB(pdb).length, 12);
  const f = createFeatures({ mode: "binder", L: 8, targetPDB: pdb });
  assert.equal(f.metadata.nTokens, 20);
  assert.equal(f.metadata.nAtoms, 160);
  assert.equal(f.metadata.atoms[112].residue, "THR");
  assert.equal(f.get("f:design_mask").data[8], 0);
  assert.equal(f.get("f:token_distance_mask").data[8 * 20 + 19], 1);
  const motif = createFeatures({
    mode: "motif",
    L: 16,
    targetPDB: pdb,
    motifIndices: Array.from({ length: 12 }, (_, i) => i + 2),
  });
  assert.equal(motif.metadata.nTokens, 16);
  assert.equal(motif.metadata.nDesigned, 4);
  assert.deepEqual(Array.from(motif.get("f:design_mask").data), [
    1,
    1,
    ...Array(12).fill(0),
    1,
    1,
  ]);
  assert.equal(new Set(motif.get("f:feature_asym_id").data).size, 1);
  assert.throws(() =>
    createFeatures({
      mode: "motif",
      L: 16,
      targetPDB: pdb,
      motifIndices: [1, 1],
    }),
  );
  const seq = "ACDEFGHIKLMNPQRSTVWY";
  const coords = Array.from({ length: seq.length * 12 }, (_, i) => i / 8);
  const custom = createFeatures({
    mode: "binder",
    L: 3,
    targetSequence: seq,
    targetCoords: coords,
  });
  assert.equal(custom.metadata.nTokens, 23);
  assert.equal(
    new Set(
      custom.metadata.atoms.filter((a) => !a.designed).map((a) => a.residue),
    ).size,
    20,
  );
  for (const L of [0, -1, 1.5, NaN]) assert.throws(() => createFeatures({ L }));
  assert.throws(() => createFeatures({ mode: "bad" }));
  assert.throws(() =>
    createFeatures({ mode: "binder", targetCoords: [1, 2, 3] }),
  );
  assert.throws(() =>
    createFeatures({
      mode: "binder",
      targetSequence: "A",
      targetCoords: [NaN],
    }),
  );
  assert.throws(() => parsePDB("ATOM"), /No protein/);
  for (const steps of [1, 3, 5, 10, 20, 50]) {
    const schedule = sigmaSchedule(steps);
    assert.equal(schedule.length, steps + 1);
    assert.equal(schedule.at(-1), 0);
    assert.ok(schedule.every(Number.isFinite));
    for (let i = 1; i < schedule.length; i++)
      assert.ok(schedule[i] < schedule[i - 1]);
    const r = randomInputs(128, steps, 41);
    assert.equal(r.size, 1 + 3 * steps);
    assert.deepEqual(r, randomInputs(128, steps, 41));
    assert.notDeepEqual(r, randomInputs(128, steps, 42));
  }
});

test("missing weights, corrupt files, and missing fixtures fail explicitly", async () => {
  const { createBoltzGenCPUEngine } = await import("../cpu/index.js");
  const read = async (path) => {
    const bytes = await readFile(path);
    return bytes.buffer.slice(
      bytes.byteOffset,
      bytes.byteOffset + bytes.byteLength,
    );
  };
  await assert.rejects(
    createBoltzGenCPUEngine({ baseUrl: root + "/weights/missing", read }),
    /ENOENT/,
  );
  await assert.rejects(
    createReader({ read: async () => new ArrayBuffer(4) })("bad", {
      byteLength: 4,
      sha256: "0".repeat(64),
    }),
    /Integrity/,
  );
  assert.throws(() => parseSafetensors(new ArrayBuffer(4)), /Invalid/);
});

test("vendor sync copies only runtime assets into an isolated local destination", async () => {
  const destination = resolve(root, ".cache/vendor-smoke");
  await vendorSync(destination);
  assert.deepEqual(
    (await readdir(destination)).sort(),
    ["LICENSE", "README_WEBGPU.md", "cpu", "shared", "webgpu"].sort(),
  );
  const module = await import(destination + "/webgpu/index.js");
  assert.equal(typeof module.createBoltzGenEngine, "function");
});

test("scratch cleanup survives command submission failure", async () => {
  const { GPUBackend } = await import("../webgpu/backend.js");
  const backend = new GPUBackend({
    addEventListener() {},
    removeEventListener() {},
    queue: {
      submit() {
        throw Error("device submission failed");
      },
    },
  });
  const destroyed = [];
  const make = (id, persistent) => ({
    bytes: 16,
    persistent,
    buffer: {
      destroy() {
        destroyed.push(id);
      },
    },
  });
  backend.live.add(make("weight", true));
  backend.live.add(make("scratch", false));
  backend.pool.set(16, [make("pooled", false)]);
  backend.encoder = {
    finish() {
      return {};
    },
  };
  assert.throws(() => backend.clearTransient(), /device submission failed/);
  assert.equal(backend.memoryUsage().transientBytes, 0);
  assert.deepEqual(destroyed.sort(), ["pooled", "scratch"]);
  backend.destroy();
  assert.equal(backend.memoryUsage().totalBytes, 0);
});
