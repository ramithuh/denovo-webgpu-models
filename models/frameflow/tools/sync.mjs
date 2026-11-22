import { cp, mkdir, access } from "node:fs/promises";
import { resolve, join } from "node:path";
import { fileURLToPath } from "node:url";
const source = fileURLToPath(new URL("../", import.meta.url));
const args = process.argv.slice(2),
  dry = args.includes("--dry-run");
const root = args.find((a) => !a.startsWith("--"));
if (!root)
  throw Error(
    "Usage: node sync-upstream.mjs /absolute/path/to/ProteinDesignSpace [--dry-run] [--weights]",
  );
const target = resolve(root, "web/static/vendor/frameflow");
await access(join(root, "web/static"));
const files = [
  "shared",
  "cpu",
  "webgpu",
  "demo",
  "LICENSE",
  "LICENSE-APACHE",
  "NOTICE",
  "README_WEBGPU.md",
  "ARCHITECTURE.md",
  "package.json",
];
if (args.includes("--weights"))
  files.push(...["fp32", "fp16", "int8"].map((p) => "weights/" + p));
console.log(JSON.stringify({ source, target, files, dryRun: dry }, null, 2));
if (!dry) {
  await mkdir(target, { recursive: true });
  for (const file of files)
    await cp(join(source, file), join(target, file), { recursive: true });
}
