import test from "node:test";
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { browserTask } from "./browser.mjs";
import { root } from "./helpers.mjs";
test(
  "real hardware WebGPU matches quantized upstream traces",
  { timeout: 3600000 },
  async () => {
    const matrix = JSON.parse(
      await readFile(root + "fixtures/mode_matrix.json", "utf8"),
    );
    assert.equal(matrix.checkpoints.length, 16);
    const variants = matrix.checkpoints
      .map((c) => c.name)
      .filter(
        (n) => !process.env.PORT_VARIANT || n === process.env.PORT_VARIANT,
      );
    const records = [];
    for (const variant of variants)
      for (const precision of ["fp32", "fp16", "int8"]) {
        const record = await browserTask(
          async (page) =>
            await page.evaluate(
              async ({ variant, precision }) => {
                const { createProtpardelleEngine } = await import(
                  "/webgpu/index.js"
                );
                const { requestHardwareAdapter } = await import(
                  "/webgpu/device.js"
                );
                const adapter = await requestHardwareAdapter();
                const info = adapter.info.toJSON?.() ?? {
                  vendor: adapter.info.vendor,
                  architecture: adapter.info.architecture,
                  device: adapter.info.device,
                  description: adapter.info.description,
                  isFallbackAdapter: adapter.info.isFallbackAdapter,
                };
                if (
                  info.isFallbackAdapter ||
                  /swiftshader|llvmpipe|software/i.test(JSON.stringify(info)) ||
                  !info.vendor
                )
                  throw new Error(`Rejected adapter ${JSON.stringify(info)}`);
                const { oracleBinary } = await import("/test/fixtures.js");
                const meta = await (
                    await fetch(`/fixtures/oracle_${variant}.json`)
                  ).json(),
                  binary = await oracleBinary(meta, "/fixtures/");
                const arrays = (entries) =>
                  Object.fromEntries(
                    Object.entries(entries).map(([k, v]) => [
                      k,
                      v === null
                        ? null
                        : new (v.dtype === "I32" ? Int32Array : Float32Array)(
                            binary.slice(v.offset, v.offset + v.length * 4),
                          ),
                    ]),
                  );
                const map = {
                  coords: "noisy_coords",
                  noiseLevel: "noise_level",
                  seqMask: "seq_mask",
                  residueIndex: "residue_index",
                  chainIndex: "chain_index",
                  hotspotMask: "hotspot_mask",
                  selfCond: "struct_self_cond",
                  cropCond: "struct_crop_cond",
                  seqCropCond: "seq_crop_cond",
                  neighborIndices: "neighbor_indices",
                };
                const start = performance.now(),
                  engine = await createProtpardelleEngine({
                    baseUrl: "/weights",
                    variant,
                    precision,
                  });
                const loadMs = performance.now() - start;
                const results = [];
                try {
                  for (const entry of meta.cases.filter(
                    (c) => c.precision === precision,
                  )) {
                    const data = arrays(entry.inputs),
                      expected = arrays(entry.traces),
                      input = Object.fromEntries(
                        Object.entries(map).map(([k, v]) => [k, data[v]]),
                      );
                    const method =
                      meta.scope === "minimpnn" ? "sequence" : "denoise";
                    const out = await engine[method](input, {
                      returnIntermediates: true,
                    });
                    for (const [name, values] of Object.entries(expected)) {
                      const actual = out.intermediates[name];
                      if (!actual || actual.length !== values.length)
                        throw new Error(`Missing trace ${name}`);
                      let error = 0,
                        norm = 0,
                        maxAbs = 0;
                      for (let i = 0; i < values.length; i++) {
                        const delta = actual[i] - values[i];
                        error += delta * delta;
                        norm += values[i] * values[i];
                        maxAbs = Math.max(maxAbs, Math.abs(delta));
                      }
                      const relRMS = Math.sqrt(error / Math.max(norm, 1e-30));
                      if (!(relRMS < 1e-4))
                        throw new Error(
                          `${variant}/${precision}/${entry.mode}/${name}: relRMS=${relRMS}, maxAbs=${maxAbs}`,
                        );
                      results.push({
                        mode: entry.mode,
                        trace: name,
                        relRMS,
                        maxAbs,
                      });
                    }
                    if (engine.memoryUsage().transientBuffers !== 0)
                      throw new Error("Transient buffer leak after success");
                    if (
                      ["unconditional", "masked_partial"].includes(entry.mode)
                    ) {
                      const warm = [];
                      for (let j = 0; j < 3; j++)
                        warm.push((await engine.denoise(input)).timingMs);
                      results.push({
                        mode: entry.mode,
                        length: entry.length,
                        warmMs: warm,
                      });
                    }
                    const controller = new AbortController();
                    const pending = engine[method](input, {
                      signal: controller.signal,
                    });
                    controller.abort();
                    let cancelled = false;
                    try {
                      await pending;
                    } catch (e) {
                      cancelled = e.name === "AbortError";
                    }
                    if (!cancelled) throw new Error("Cancellation failed");
                    if (engine.memoryUsage().transientBuffers !== 0)
                      throw new Error(
                        "Transient buffer leak after cancellation",
                      );
                  }
                } finally {
                  engine.destroy();
                  engine.destroy();
                }
                if (engine.memoryUsage().bytes !== 0)
                  throw new Error("Resident buffer leak");
                return { variant, precision, adapter: info, loadMs, results };
              },
              { variant, precision },
            ),
        );
        records.push(record);
        console.log(`${variant}/${precision}: hardware traces pass`);
      }
    assert.ok(records.length);
    await writeFile(
      root +
        `fixtures/gpu_metrics${process.env.PORT_VARIANT ? "_" + process.env.PORT_VARIANT : ""}.json`,
      JSON.stringify(records, null, 2) + "\n",
    );
  },
);
