// Explicit destination is mandatory. No default writes into sibling projects.
import { cp, mkdir } from "node:fs/promises";
import { resolve } from "node:path";
export async function vendorSync(destination) {
  if (!destination) throw Error("Specify a destination directory");
  const root = resolve(import.meta.dirname, ".."),
    target = resolve(destination);
  await mkdir(target, { recursive: true });
  for (const name of ["shared", "cpu", "webgpu", "LICENSE", "README_WEBGPU.md"])
    await cp(resolve(root, name), resolve(target, name), { recursive: true });
}
if (process.argv[1] === new URL(import.meta.url).pathname)
  await vendorSync(process.argv[2]);
