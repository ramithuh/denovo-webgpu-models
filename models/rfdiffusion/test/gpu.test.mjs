import test from "node:test";
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { browserTask } from "./browser.mjs";
import { root } from "./helpers.mjs";
test(
  "real hardware WGSL denoiser and SE(3) trajectories, all modes and precisions",
  { timeout: 3600000 },
  async () => {
    const result = await browserTask((page) =>
      page.evaluate(async () => {
        const { createRFdiffusionEngine } = await import("/webgpu/index.js"),
          { requestHardwareAdapter } = await import("/webgpu/device.js"),
          { readFixture, readTrajectory, verifyDenoiser, verifyTrajectory } =
            await import("/test/parity.js"),
          adapter = await requestHardwareAdapter(),
          info = {
            vendor: adapter.info.vendor,
            architecture: adapter.info.architecture,
            device: adapter.info.device,
            description: adapter.info.description,
            isFallbackAdapter: adapter.info.isFallbackAdapter,
          };
        const read = async (url) => {
            const r = await fetch(url);
            if (!r.ok) throw Error(`Missing fixture ${url}`);
            return r.arrayBuffer();
          },
          matrix = JSON.parse(
            new TextDecoder().decode(await read("/fixtures/mode_matrix.json")),
          ),
          records = [];
        if (matrix.cases.length !== 12 || matrix.trajectories.length !== 9)
          throw Error("Incomplete mode matrix");
        for (const precision of ["fp32", "fp16", "int8"]) {
          const start = performance.now(),
            engine = await createRFdiffusionEngine({
              baseUrl: "/weights",
              precision,
              cache: "none",
            }),
            loadMs = performance.now() - start;
          try {
            for (const c of matrix.cases.filter(
              (c) => c.precision === precision,
            )) {
              const f = await readFixture(c.name, read);
              records.push({ ...(await verifyDenoiser(engine, f)), loadMs });
              if (engine.memoryUsage().transientBuffers !== 0)
                throw Error("Transient leak after denoising");
              if (c.mode === "unconditional" && !c.selfcond) {
                const controller = new AbortController();
                let cancelled = false;
                try {
                  await engine.denoiseStep(f.input, {
                    signal: controller.signal,
                    onProgress: () => controller.abort(),
                  });
                } catch (e) {
                  if (e.name !== "AbortError") throw e;
                  cancelled = true;
                }
                if (!cancelled || engine.memoryUsage().transientBuffers !== 0)
                  throw Error("Abort hygiene failed");
                let thrown = false;
                try {
                  await engine.denoiseStep(f.input, {
                    onProgress: () => {
                      throw Error("injected");
                    },
                  });
                } catch (e) {
                  if (e.message !== "injected") throw e;
                  thrown = true;
                }
                if (!thrown || engine.memoryUsage().transientBuffers !== 0)
                  throw Error("Error hygiene failed");
              }
            }
            for (const name of matrix.trajectories.filter((n) =>
              n.includes("_" + precision + "_"),
            )) {
              records.push(
                await verifyTrajectory(
                  engine,
                  await readTrajectory(name, read),
                ),
              );
              if (engine.memoryUsage().transientBuffers !== 0)
                throw Error("Transient leak after sampling");
            }
          } finally {
            engine.destroy();
            engine.destroy();
          }
          if (engine.memoryUsage().bytes !== 0)
            throw Error("Leak after destroy");
        }
        return { adapter: info, records };
      }),
    );
    assert.equal(result.adapter.vendor, "nvidia");
    await writeFile(
      root + "fixtures/gpu_metrics.json",
      JSON.stringify(result, null, 2) + "\n",
    );
  },
);
