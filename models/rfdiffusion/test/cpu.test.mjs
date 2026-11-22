import test from "node:test";
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { Worker } from "node:worker_threads";
import { root } from "./helpers.mjs";
test(
  "complete CPU denoiser and three-step SE(3) trajectories, all modes and precisions",
  { timeout: 3600000 },
  async () => {
    const matrix = JSON.parse(
      await readFile(root + "fixtures/mode_matrix.json"),
    );
    assert.equal(matrix.cases.length, 12);
    assert.equal(matrix.trajectories.length, 9);
    const workers = [];
    try {
      const batches = await Promise.all(
        [
          ...["fp32", "fp16", "int8"].map((precision) => ({ precision })),
          { precision: "fp32", stress: true },
        ].map(
          (task) =>
            new Promise((resolve, reject) => {
              const worker = new Worker(
                new URL("./cpu-runner.mjs", import.meta.url),
                { workerData: { ...task, matrix } },
              );
              workers.push(worker);
              let done = false;
              worker.on("message", (message) => {
                if (message.progress) console.log("CPU", message.progress);
                else if (message.records) {
                  done = true;
                  resolve(message.records);
                }
              });
              worker.on("error", reject);
              worker.on("exit", (code) => {
                if (code !== 0 || !done)
                  reject(
                    Error(
                      "CPU fixture worker exited without successful results: " +
                        code,
                    ),
                  );
              });
            }),
        ),
      );
      const records = batches.flat();
      assert.equal(records.length, 22);
      await writeFile(
        root + "fixtures/cpu_metrics.json",
        JSON.stringify(records, null, 2) + "\n",
      );
    } finally {
      await Promise.allSettled(workers.map((worker) => worker.terminate()));
    }
  },
);
