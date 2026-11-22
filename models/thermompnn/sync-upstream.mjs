#!/usr/bin/env node
/** Copy the runtime vendor contract only. Dry-run is the default. */
import { readdir, readFile, writeFile, mkdir, lstat } from "node:fs/promises";
import { resolve, dirname, sep } from "node:path";
import { fileURLToPath } from "node:url";
const root = fileURLToPath(new URL("./", import.meta.url));
const args = process.argv.slice(2),
  destination = args.find((a) => !a.startsWith("--"));
if (
  !destination ||
  args.some((a) => a.startsWith("--") && !["--write", "--no-cpu"].includes(a))
)
  throw new Error(
    "Usage: node sync-upstream.mjs DESTINATION [--write] [--no-cpu]",
  );
const target = resolve(destination);
if (target === resolve(root) || resolve(root).startsWith(target + sep))
  throw new Error(
    "Destination cannot replace the source repository or its ancestor",
  );
const files = ["LICENSE", "README_WEBGPU.md"];
for (const dir of [
  "shared",
  "webgpu",
  ...(args.includes("--no-cpu") ? [] : ["cpu"]),
]) {
  for (const e of await readdir(resolve(root, dir), { withFileTypes: true })) {
    if (e.isSymbolicLink() || !e.isFile() || !e.name.endsWith(".js"))
      throw new Error(`Unexpected runtime entry: ${dir}/${e.name}`);
    files.push(dir + "/" + e.name);
  }
}
for (const file of files)
  console.log(
    `${args.includes("--write") ? "COPY" : "DRY RUN"} ${file} -> ${resolve(target, file)}`,
  );
if (args.includes("--write")) {
  // Reject symlink destinations before following a path outside the requested tree.
  for (let p = target; p !== dirname(p); p = dirname(p)) {
    try {
      if ((await lstat(p)).isSymbolicLink())
        throw new Error(`Symlink destination: ${p}`);
    } catch (e) {
      if (e.code !== "ENOENT") throw e;
    }
  }
  for (const file of files) {
    const out = resolve(target, file);
    for (
      let parent = dirname(out);
      parent !== target;
      parent = dirname(parent)
    ) {
      try {
        if ((await lstat(parent)).isSymbolicLink())
          throw new Error(`Symlink directory: ${parent}`);
      } catch (error) {
        if (error.code !== "ENOENT") throw error;
      }
    }
    await mkdir(dirname(out), { recursive: true });
    try {
      if ((await lstat(out)).isSymbolicLink())
        throw new Error(`Symlink file: ${out}`);
    } catch (e) {
      if (e.code !== "ENOENT") throw e;
    }
    await writeFile(out, await readFile(resolve(root, file)));
  }
  console.log(`Copied ${files.length} runtime/documentation files.`);
}
