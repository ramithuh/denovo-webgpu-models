import { spawn } from "node:child_process";
import { createWriteStream } from "node:fs";
import { mkdir, writeFile, readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
const root = fileURLToPath(new URL("../", import.meta.url));
await mkdir(root + ".cache/validation", { recursive: true });
const records = [];
const selected = process.argv[2];
if (selected && !["test", "test:cpu", "test:gpu"].includes(selected))
  throw new Error("Optional argument must be test, test:cpu, or test:gpu");
async function run(args) {
  const command = "npm " + args.join(" "),
    file = root + ".cache/validation/" + args.at(-1).replace(":", "-") + ".log",
    log = createWriteStream(file),
    start = Date.now();
  console.log("START " + command);
  const env = { ...process.env };
  delete env.PORT_VARIANT;
  delete env.PORT_PRESET;
  const child = spawn("npm", args, {
    cwd: root,
    env,
    stdio: ["ignore", "pipe", "pipe"],
  });
  child.stdout.pipe(log);
  child.stderr.pipe(log);
  const code = await new Promise((resolve) => child.on("exit", resolve));
  log.end();
  const record = {
    command,
    exitCode: code,
    elapsedSeconds: (Date.now() - start) / 1000,
    completedAt: new Date().toISOString(),
  };
  if (selected) {
    try {
      const previous = JSON.parse(
        await readFile(root + "fixtures/validation_commands.json", "utf8"),
      );
      records.push(...previous.filter((entry) => entry.command !== command));
    } catch (error) {
      if (error.code !== "ENOENT") throw error;
    }
  }
  records.push(record);
  console.log(JSON.stringify(record));
  await writeFile(
    root + "fixtures/validation_commands.json",
    JSON.stringify(records, null, 2) + "\n",
  );
  if (code !== 0) throw new Error(command + " failed; see " + file);
}
if (selected) {
  await run(selected === "test" ? ["test"] : ["run", selected]);
} else {
  // These suites write distinct metric files; only the GPU suite owns a browser.
  const remaining = await Promise.allSettled([
    run(["run", "test:cpu"]),
    run(["run", "test:gpu"]),
  ]);
  for (const result of remaining)
    if (result.status === "rejected") throw result.reason;
  await run(["test"]);
}
