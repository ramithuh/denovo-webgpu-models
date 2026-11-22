import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { browserTask } from "./browser.mjs";
import { root } from "./helpers.mjs";
import { parseStructure } from "../shared/structure.js";
test(
  "static demo generates, downloads and releases hardware resources",
  { timeout: 120000 },
  async () => {
    await browserTask(async (page, url) => {
      await page.goto(url + "/demo/index.html");
      await page.locator("#length").fill("8");
      await page.locator("#steps").fill("4");
      await page.locator("#generate").click();
      await page.waitForFunction(
        () =>
          document.querySelector("#status").textContent ===
          "Generation complete.",
        {},
        { timeout: 90000 },
      );
      assert.equal(await page.locator("#residues").textContent(), "8");
      const download = page.waitForEvent("download");
      await page.locator("#pdb").click();
      const pdb = await download;
      assert.equal(pdb.suggestedFilename(), "protpardelle.pdb");
      await pdb.saveAs(root + ".cache/demo-result.pdb");
      assert.equal(
        parseStructure(await readFile(root + ".cache/demo-result.pdb", "utf8"))
          .aatype.length,
        8,
      );
      await page.screenshot({ path: root + ".cache/demo.png", fullPage: true });
      await page.locator("#release").click();
      await page.waitForFunction(
        () => document.querySelector("#memory").textContent === "0 MB",
      );
    });
  },
);
