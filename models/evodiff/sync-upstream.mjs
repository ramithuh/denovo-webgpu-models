#!/usr/bin/env node
import { cp, mkdir, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
const root = fileURLToPath(new URL(".", import.meta.url));
const args = process.argv.slice(2),
  write = args.includes("--write"),
  targetArg = args.find((x) => !x.startsWith("--"));
if (!targetArg || args.some((x) => x.startsWith("--") && x !== "--write"))
  throw new Error("Usage: node sync-upstream.mjs TARGET [--write]");
const target = path.resolve(targetArg);
if (
  target === path.resolve(root) ||
  ["shared", "cpu", "webgpu"].some((p) => target.startsWith(path.join(root, p)))
)
  throw new Error("Vendor target overlaps runtime source");
const items = ["shared", "cpu", "webgpu", "LICENSE", "README_WEBGPU.md"];
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
