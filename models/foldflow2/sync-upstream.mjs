#!/usr/bin/env node
/** Runtime-only vendor copy. Dry-run unless --write is explicitly supplied. */
import {
  readdir,
  readFile,
  mkdir,
  writeFile,
  realpath,
  lstat,
} from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
const root = fileURLToPath(new URL("./", import.meta.url)),
  args = process.argv.slice(2),
  write = args.includes("--write"),
  destination = args.find((a) => !a.startsWith("--"));
if (!destination || args.some((a) => a.startsWith("--") && a !== "--write"))
  throw new Error("Usage: node sync-upstream.mjs DEST [--write]");
const dest = path.resolve(destination, "web/static/vendor/foldflow2"),
  allowed = [
    "shared",
    "webgpu",
    "cpu",
    "LICENSE",
    "THIRD_PARTY_LICENSES",
    "README_WEBGPU.md",
  ],
  files = [];
async function collect(relative) {
  const p = path.join(root, relative),
    stat = await lstat(p);
  if (stat.isSymbolicLink()) throw new Error("Source symlinks forbidden");
  if (stat.isDirectory())
    for (const f of await readdir(p)) await collect(path.join(relative, f));
  else files.push(relative);
}
for (const entry of allowed) await collect(entry);
for (const file of files)
  console.log(
    `${write ? "COPY" : "WOULD COPY"} ${file} -> ${path.join(dest, file)}`,
  );
async function safeDirectory(directory) {
  let ancestor = directory;
  while (true) {
    try {
      const canonical = await realpath(ancestor);
      if (canonical !== ancestor)
        throw new Error("Destination symlink forbidden");
      break;
    } catch (e) {
      if (e.code !== "ENOENT") throw e;
      ancestor = path.dirname(ancestor);
    }
  }
  await mkdir(directory, { recursive: true });
}
if (write) {
  await safeDirectory(dest);
  for (const file of files) {
    const target = path.join(dest, file);
    await safeDirectory(path.dirname(target));
    try {
      if ((await lstat(target)).isSymbolicLink())
        throw new Error("Destination symlink forbidden");
    } catch (e) {
      if (e.code !== "ENOENT") throw e;
    }
    await writeFile(target, await readFile(path.join(root, file)));
  }
}
