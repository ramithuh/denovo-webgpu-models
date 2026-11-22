import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import { root } from "./helpers.mjs";
test("shared and webgpu runtime modules contain no Node-only imports", async () => {
  async function walk(url) {
    for (const f of await fs.readdir(url, { withFileTypes: true })) {
      const path = new URL(f.name + (f.isDirectory() ? "/" : ""), url);
      if (f.isDirectory()) await walk(path);
      else if (f.name.endsWith(".js")) {
        const text = await fs.readFile(path, "utf8");
        assert.doesNotMatch(
          text,
          /(?:from\s*|import\s*\(|import\s*)['"](?:node:|(?:fs|path|url|buffer|crypto)(?:\/|['"]))/i,
          path.href,
        );
      }
    }
  }
  await walk(new URL("shared/", root));
  await walk(new URL("webgpu/", root));
});
