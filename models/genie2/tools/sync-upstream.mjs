// Explicit promotion utility. Default is a dry run; never invoked by bootstrap/tests.
import { cp, mkdir, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
const root = fileURLToPath(new URL("../", import.meta.url));
const args = process.argv.slice(2),
  idx = args.indexOf("--target");
const target = path.resolve(
  idx >= 0
    ? args[idx + 1]
    : path.join(root, "../../../ProteinDesignSpace/web/static/vendor/genie2"),
);
if (!target.endsWith(path.join("web", "static", "vendor", "genie2")))
  throw Error("Destination must end in web/static/vendor/genie2");
const files = [
  "shared",
  "cpu",
  "webgpu",
  "demo",
  "LICENSE",
  "NOTICE",
  "THIRD_PARTY_LICENSES.txt",
  "README_WEBGPU.md",
  "package.json",
];
for (const name of files) await stat(path.join(root, name));
console.log(
  JSON.stringify(
    { source: root, target, files, write: args.includes("--write") },
    null,
    2,
  ),
);
if (args.includes("--write")) {
  await mkdir(target, { recursive: true });
  for (const name of files)
    await cp(path.join(root, name), path.join(target, name), {
      recursive: true,
    });
}
