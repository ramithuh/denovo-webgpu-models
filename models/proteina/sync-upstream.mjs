#!/usr/bin/env node
// Dry run by default. The coordinator runs --write after promotion approval.
import { cp, mkdir, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
const root = fileURLToPath(new URL(".", import.meta.url)),
  args = process.argv.slice(2),
  write = args.includes("--write"),
  targetArg = args.find((x) => !x.startsWith("--"));
if (args.some((x) => x.startsWith("--") && x !== "--write"))
  throw Error(
    "Usage: node sync-upstream.mjs [ProteinDesignSpace-root] [--write]",
  );
const project = path.resolve(
    targetArg ?? path.join(root, "../../../ProteinDesignSpace"),
  ),
  target = path.join(project, "web/static/vendor/proteina");
if (target.startsWith(path.resolve(root) + path.sep))
  throw Error("Vendor destination cannot overlap the source package");
const items = [
  "shared",
  "cpu",
  "webgpu",
  "demo",
  "README_WEBGPU.md",
  "NOTICE.md",
  "LICENSE.NVIDIA",
  "LICENSE.OpenFold",
  "LICENSE.PairBiasAttention",
  "LICENSE.AF3Utils",
];
for (const item of items) await stat(path.join(root, item));
if (write) {
  await mkdir(target, { recursive: true });
  for (const item of items)
    await cp(path.join(root, item), path.join(target, item), {
      recursive: true,
    });
}
console.log(
  `${write ? "Copied" : "Would copy"} ${items.join(", ")} to ${target}`,
);
