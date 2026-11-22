import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { createHash } from "node:crypto";
import { decodeSafetensors } from "../shared/ligandmpnn/weights.js";
for (const variant of ["010", "020"])
  for (const precision of ["fp32", "fp16", "int8"])
    test(`Bundle integrity ${variant}/${precision}`, () => {
      const base = new URL(
          `../weights/ligandmpnn_v_32_${variant}_25/${precision}/`,
          import.meta.url,
        ),
        m = JSON.parse(fs.readFileSync(new URL("manifest.json", base)));
      assert.equal(m.num_tensors, 197);
      assert.equal(m.total_params, 2618693);
      const shards = {};
      for (const s of m.shards) {
        const data = fs.readFileSync(new URL(s.file, base));
        assert.ok(data.byteLength <= 25_000_000);
        assert.equal(data.byteLength, s.bytes);
        assert.equal(createHash("sha256").update(data).digest("hex"), s.sha256);
        shards[s.file] = decodeSafetensors(
          data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength),
        );
      }
      for (const [name, r] of Object.entries(m.tensors)) {
        assert.deepEqual(shards[r.shard][name].shape, r.shape);
        assert.equal(
          r.dtype,
          r.shape.length === 1 || precision === "fp32"
            ? "F32"
            : precision === "fp16"
              ? "F16"
              : "I8",
        );
        if (r.quant) {
          assert.equal(r.quant.axis, 0);
          assert.ok(
            shards[r.shard][r.quant.scale_tensor].data.every((v) => v > 0),
          );
        }
      }
    });
