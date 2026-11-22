import { cp, mkdir, readdir, lstat } from "node:fs/promises";
import { resolve, relative, join } from "node:path";
import { fileURLToPath } from "node:url";
const root = fileURLToPath(new URL("./", import.meta.url)),
  args = process.argv.slice(2),
  write = args.includes("--write"),
  destination = args.find((a) => !a.startsWith("--"));
if (!destination)
  throw new Error("Usage: node sync-upstream.mjs DESTINATION [--write]");
const dest = resolve(destination);
if (
  dest === resolve(root) ||
  (!relative(root, dest).startsWith("..") &&
    !relative(root, dest).startsWith(".cache/"))
)
  throw new Error("Destination must not overwrite source files");
const files = [];
async function collect(path) {
  const stat = await lstat(join(root, path));
  if (stat.isSymbolicLink()) throw new Error("Symlinks are not vendor inputs");
  if (stat.isDirectory())
    for (const name of (await readdir(join(root, path))).sort())
      await collect(join(path, name));
  else files.push(path);
}
for (const name of ["shared", "cpu", "webgpu", "LICENSE", "README_WEBGPU.md"])
  await collect(name);
for (const path of files) {
  console.log(`${write ? "COPY" : "PLAN"} ${path}`);
  if (write) {
    await mkdir(resolve(dest, path, ".."), { recursive: true });
    await cp(join(root, path), join(dest, path));
  }
}
console.log(
  `${write ? "Copied" : "Would copy"} ${files.length} runtime/license/documentation files to ${dest}`,
);
