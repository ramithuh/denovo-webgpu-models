#!/usr/bin/env node
/** Dry-run by default. This task only stages a reviewable vendor tree inside the package. */
import { readdir, mkdir, copyFile, realpath, lstat } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
const root = fileURLToPath(new URL("../", import.meta.url)),
  args = process.argv.slice(2),
  write = args.includes("--write"),
  destination =
    args.find((a) => !a.startsWith("--")) ??
    ".cache/vendor-stage/ProteinDesignSpace/web/static/vendor/sid-protein";
const target = path.resolve(root, destination),
  relative = path.relative(root, target);
if (
  relative.startsWith("..") ||
  path.isAbsolute(relative) ||
  relative === "" ||
  !relative.startsWith(".cache/")
)
  throw new Error(
    "This isolated task can only stage inside models/sid-protein/.cache/; the coordinator promotes the vendor tree.",
  );
const files = [
  "README_WEBGPU.md",
  "LICENSE_SID",
  "LICENSE_PROTEINA",
  "LICENSE_BOOST",
  "NOTICE",
  "fixtures/export_metrics.json",
  "fixtures/gpu_metrics.json",
  "fixtures/cpu_metrics.json",
  "fixtures/validation.json",
  "fixtures/mode_matrix.json",
  "fixtures/demo_metrics.json",
  "fixtures/schedule_metrics.json",
];
async function collect(dir) {
  for (const entry of await readdir(path.join(root, dir), {
    withFileTypes: true,
  })) {
    if (entry.isSymbolicLink()) throw new Error("Symlink not vendorable");
    if (entry.isDirectory()) await collect(path.join(dir, entry.name));
    else if (/\.(js|mjs|wgsl|html|css)$/.test(entry.name))
      files.push(path.join(dir, entry.name));
  }
}
for (const dir of ["shared", "cpu", "webgpu", "demo"]) await collect(dir);
for (const file of files) {
  if (write) {
    const dest = path.join(target, file);
    await mkdir(path.dirname(dest), { recursive: true });
    const actual = await realpath(path.dirname(dest));
    if (!actual.startsWith(root))
      throw new Error("Destination escaped package");
    try {
      if ((await lstat(dest)).isSymbolicLink())
        throw new Error("Destination symlink");
    } catch (e) {
      if (e.code !== "ENOENT") throw e;
    }
    await copyFile(path.join(root, file), dest);
  }
}
console.log(
  JSON.stringify(
    {
      write,
      target,
      coordinatorDestination:
        "ProteinDesignSpace/web/static/vendor/sid-protein/",
      files,
    },
    null,
    2,
  ),
);
