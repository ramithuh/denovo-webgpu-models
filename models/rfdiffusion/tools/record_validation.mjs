#!/usr/bin/env node
// Record only completed TAP runs with every test passing and zero skips.
import { readFile, writeFile, readdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
const root = fileURLToPath(new URL("../", import.meta.url));
const commands = [];
for (const [command, file] of [
  ["npm test", "final-npm-test.log"],
  ["npm run test:cpu", "final-test-cpu.log"],
  ["npm run test:gpu", "final-test-gpu.log"],
  ["node --test test/benchmark.test.mjs", "final-benchmark.log"],
]) {
  const text = await readFile(root + ".cache/" + file, "utf8");
  const value = (k) => {
    const m = [
      ...text.matchAll(new RegExp("^# " + k + " ([0-9.]+)$", "gm")),
    ].at(-1);
    if (!m) throw Error("Missing TAP result " + k + " in " + file);
    return Number(m[1]);
  };
  const tests = value("tests"),
    pass = value("pass"),
    fail = value("fail"),
    skipped = value("skipped"),
    cancelled = value("cancelled"),
    todo = value("todo");
  if (!tests || pass !== tests || fail || skipped || cancelled || todo)
    throw Error("Incomplete validation: " + command);
  commands.push({
    command,
    tests,
    pass,
    fail,
    skipped,
    cancelled,
    todo,
    durationMs: value("duration_ms"),
    log: ".cache/" + file,
    logSha256: createHash("sha256").update(text).digest("hex"),
  });
}
const hash = createHash("sha256"),
  files = [];
for (const dir of ["shared", "cpu", "webgpu"])
  for (const file of (await readdir(root + dir)).sort())
    if (/\.(js|mjs|wgsl)$/.test(file)) {
      const name = dir + "/" + file;
      hash.update(name + "\0");
      hash.update(await readFile(root + name));
      files.push(name);
    }
const runtimeSourceSha256 = hash.digest("hex");
const testedSource = (
  await readFile(root + ".cache/runtime-before-tests.sha256", "utf8")
).trim();
if (runtimeSourceSha256 !== testedSource)
  throw Error("Runtime changed after final validation started");
const result = {
  generatedAt: new Date().toISOString(),
  upstreamCommit: "86507b6538f51fce57b5a72477165f03999ed7ae",
  node: process.version,
  adapter: JSON.parse(
    await readFile(root + "fixtures/gpu_metrics.json", "utf8"),
  ).adapter,
  runtimeSourceSha256,
  runtimeFiles: files,
  commands,
};
await writeFile(
  root + "fixtures/validation.json",
  JSON.stringify(result, null, 2) + "\n",
);
console.log(JSON.stringify(result.commands, null, 2));
