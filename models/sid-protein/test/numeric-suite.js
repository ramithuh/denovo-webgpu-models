import { GpuBackend } from "../webgpu/index.js";
import { acquireDevice } from "../webgpu/device.js";
import { diffStats } from "../shared/index.js";
export async function run() {
  const b = new GpuBackend(
    await acquireDevice({ allowSoftwareFallback: false }),
  );
  const o = await (await fetch("/fixtures/operators.json")).json(),
    get = async (r) =>
      new Float32Array(await (await fetch("/" + r.file)).arrayBuffer()),
    rows = [];
  try {
    for (const c of o.norms) {
      for (const [n, r] of [
        ["norm.weight", c.weight],
        ["norm.bias", c.bias],
      ]) {
        const a = await get(r);
        await b.upload(n, {
          dtype: "F32",
          shape: [c.width],
          bytes: new Uint8Array(a.buffer),
        });
      }
      b.begin();
      try {
        const x = b.tensor(await get(c.input), 3, c.width),
          y = await b.read(await b.norm(x, "norm"));
        const ref = await get(c.output);
        rows.push({
          name: "norm" + c.width,
          stats: diffStats(y, ref),
          exact: y.filter((v, i) => v === ref[i]).length / y.length,
          samples: [...y.slice(0, 5)],
          ref: [...ref.slice(0, 5)],
        });
      } finally {
        await b.end();
      }
    }
    b.begin();
    try {
      for (const [name, op] of [
        ["time", () => b.time(Math.fround(0.37))],
        ["position", () => b.position(32)],
      ]) {
        const y = await b.read(await op()),
          ref = await get(o[name]);
        rows.push({
          name,
          stats: diffStats(y, ref),
          exact: y.filter((v, i) => v === ref[i]).length / y.length,
        });
      }
    } finally {
      await b.end();
    }
    if (rows.some((r) => r.stats.relRms >= 1e-6))
      throw new Error("Numeric operator parity failed " + JSON.stringify(rows));
    return { rows };
  } finally {
    b.destroy();
  }
}
