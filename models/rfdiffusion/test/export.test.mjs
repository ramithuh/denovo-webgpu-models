import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { root, localRead } from "./helpers.mjs";
import { loadBundle, parseSafetensors, sha256 } from "../shared/loader.js";
test("all 5998 checkpoint tensors export with verified bounded shards and colocated scales", async () => {
  for (const precision of ["fp32", "fp16", "int8"]) {
    const base = root + "weights/base/" + precision,
      m = JSON.parse(await readFile(base + "/manifest.json"));
    assert.equal(m.format, "rfdiffusion-safetensors-v1");
    assert.equal(m.upstream_commit, "86507b6538f51fce57b5a72477165f03999ed7ae");
    assert.equal(Object.keys(m.tensors).length, 5998);
    let sum = 0;
    for (const s of m.shards) {
      const bin = await localRead(base + "/" + s.file);
      assert.equal(bin.byteLength, s.byteLength);
      assert.ok(s.byteLength <= 25000000);
      assert.equal(await sha256(bin), s.sha256);
      const tensors = parseSafetensors(bin);
      for (const [name, t] of Object.entries(tensors))
        if (t.dtype === "I8") {
          assert.equal(tensors[name + ".__scale"].dtype, "F32");
          assert.equal(tensors[name + ".__scale"].shape[0], t.shape[0]);
        }
      sum += s.byteLength;
    }
    assert.equal(sum, m.total_bytes);
  }
});
test("missing or corrupted weights fail", async () => {
  await assert.rejects(
    loadBundle({ baseUrl: root + ".cache/missing", read: localRead }),
  );
  await assert.rejects(
    loadBundle({
      baseUrl: root + "weights/base/int8",
      read: async (url) => {
        const b = await localRead(url);
        if (url.endsWith(".safetensors"))
          new Uint8Array(b)[b.byteLength - 1] ^= 1;
        return b;
      },
    }),
    /integrity/,
  );
});
