import { Worker } from "node:worker_threads";
import { writeFile } from "node:fs/promises";
import { root } from "./io.mjs";
const start = performance.now(),
  workers = [];
try {
  const result = await Promise.all(
    [[1], [0, 4], [3, 2, 5]].map(
      (indices) =>
        new Promise((resolve, reject) => {
          const worker = new Worker(
            new URL("./cpu-worker.mjs", import.meta.url),
            { workerData: { indices } },
          );
          workers.push(worker);
          let done;
          worker.on("message", (msg) => {
            if (msg.kind === "progress")
              console.log(`${msg.name}: ${msg.step}/${msg.steps}`);
            else if (msg.kind === "row") console.log(JSON.stringify(msg.row));
            else if (msg.kind === "done") done = msg;
          });
          worker.on("error", reject);
          worker.on("exit", (code) => {
            if (code || !done) reject(new Error(`CPU worker failed (${code})`));
            else resolve(done);
          });
        }),
    ),
  );
  const metrics = {
    ok: true,
    skipped: 0,
    rows: result.flatMap((r) => r.rows),
    milliseconds: performance.now() - start,
    memoryAfterDestroy: result.map((r) => r.memoryAfterDestroy),
  };
  if (metrics.memoryAfterDestroy.some((m) => m.totalBytes !== 0))
    throw new Error("CPU destroy leaked weights");
  await writeFile(
    new URL("fixtures/cpu_metrics.json", root),
    JSON.stringify(metrics, null, 2) + "\n",
  );
  console.log("All CPU forwards and complete trajectories passed; zero skips.");
} finally {
  await Promise.all(workers.map((w) => w.terminate()));
}
