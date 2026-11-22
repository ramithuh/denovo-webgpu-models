import { mkdir, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
export async function buildLoaderFixture() {
  const root = new URL("../.cache/loader/", import.meta.url);
  await mkdir(root, { recursive: true });
  const m = {
    format: "sid-protein-safetensors-v1",
    precision: "int8",
    config: {},
    num_tensors: 1,
    total_params: 12,
    tensors: { "test.weight": { shape: [4, 3], dtype: "I8", parts: [] } },
    shards: [],
  };
  for (let i = 0; i < 2; i++) {
    const key = "part" + i,
      skey = key + "__scale",
      header = {
        [key]: { dtype: "I8", shape: [2, 3], data_offsets: [0, 6] },
        [skey]: { dtype: "F32", shape: [2], data_offsets: [6, 14] },
      };
    let h = JSON.stringify(header);
    h = h.padEnd(Math.ceil(h.length / 8) * 8, " ");
    const bytes = Buffer.alloc(8 + h.length + 14);
    bytes.writeBigUInt64LE(BigInt(h.length));
    bytes.write(h, 8);
    const q = i === 0 ? [1, 2, 3, -2, -1, 0] : [3, 0, -3, 7, 8, 9];
    for (let j = 0; j < 6; j++) bytes.writeInt8(q[j], 8 + h.length + j);
    bytes.writeFloatLE(i === 0 ? 1 : 0.5, 8 + h.length + 6);
    bytes.writeFloatLE(1, 8 + h.length + 10);
    const file = `part${i}.safetensors`;
    await writeFile(new URL(file, root), bytes);
    m.shards.push({
      file,
      bytes: bytes.length,
      byteLength: bytes.length,
      sha256: createHash("sha256").update(bytes).digest("hex"),
    });
    m.tensors["test.weight"].parts.push({
      key,
      shard: file,
      shape: [2, 3],
      row_start: i * 2,
      dtype: "I8",
      quant: { scheme: "symmetric_per_row", axis: 0, scale_tensor: skey },
    });
  }
  m.total_bytes = m.shards.reduce((n, s) => n + s.bytes, 0);
  await writeFile(new URL("manifest.json", root), JSON.stringify(m));
}
