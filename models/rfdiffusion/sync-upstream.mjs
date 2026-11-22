#!/usr/bin/env node
import { readdir, mkdir, copyFile, lstat } from "node:fs/promises";
import { resolve, relative, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
const source = dirname(fileURLToPath(import.meta.url)),
  args = process.argv.slice(2),
  write = args.includes("--write"),
  targetArg = args.find((x) => !x.startsWith("--"));
if (!targetArg) throw Error("Usage: node sync-upstream.mjs [--write] TARGET");
const target = resolve(targetArg);
if (
  target === source ||
  target.startsWith(source + "/shared") ||
  target.startsWith(source + "/cpu") ||
  target.startsWith(source + "/webgpu")
)
  throw Error("Target overlaps runtime source");
const files = ["LICENSE", "README_WEBGPU.md"];
async function list(dir) {
  for (const item of await readdir(join(source, dir), {
    withFileTypes: true,
  })) {
    if (item.isSymbolicLink())
      throw Error("Runtime symlinks are not vendorable");
    if (item.isDirectory()) await list(join(dir, item.name));
    else if (/\.(?:js|mjs|wgsl)$/.test(item.name))
      files.push(join(dir, item.name));
  }
}
for (const dir of ["shared", "cpu", "webgpu"]) await list(dir);
for (const file of files) {
  const destination = join(target, file);
  if (write) {
    await mkdir(dirname(destination), { recursive: true });
    await copyFile(join(source, file), destination);
  }
}
console.log(JSON.stringify({ write, target, files }, null, 2));
