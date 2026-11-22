#!/usr/bin/env node
/** Produce a fresh vendorable tree without touching an application checkout. */
import { cp, mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
const source = fileURLToPath(new URL("../", import.meta.url));
const index = process.argv.indexOf("--out");
if (index < 0 || !process.argv[index + 1])
  throw Error(
    "Usage: node tools/sync-upstream.mjs --out <new-staging-directory>",
  );
const target = resolve(process.argv[index + 1]);
if (target === source || source.startsWith(target + "/"))
  throw Error("Destination may not contain the source checkout");
await mkdir(dirname(target), { recursive: true });
await mkdir(target); // Existing destinations are intentionally never overwritten.
for (const name of [
  "shared",
  "cpu",
  "webgpu",
  "LICENSE",
  "README_WEBGPU.md",
  "package.json",
])
  await cp(resolve(source, name), resolve(target, name), { recursive: true });
const pkg = JSON.parse(await readFile(resolve(target, "package.json"), "utf8"));
delete pkg.scripts;
await writeFile(
  resolve(target, "package.json"),
  JSON.stringify(pkg, null, 2) + "\n",
);
console.log(
  `Vendored runtime staged at ${target}. Weights are loaded separately by URL.`,
);
