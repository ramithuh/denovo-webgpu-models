import test from "node:test";
import assert from "node:assert/strict";
import { browserTask } from "./browser.mjs";
import { root } from "./helpers.mjs";
test(
  "static demo generates a third dynamic length and downloadable full-atom PDB",
  { timeout: 120000 },
  async () => {
    const result = await browserTask(async (page, url) => {
      await page.goto(url + "/demo/index.html");
      await page.locator("#length").fill("17");
      await page.locator("#steps").fill("2");
      await page.locator("#precision").selectOption("int8");
      await page.locator("#run").click();
      await page
        .locator("#download")
        .waitFor({ state: "visible", timeout: 60000 });
      const text = await page.locator("#status").textContent();
      if (!/nvidia/i.test(text))
        throw Error("Demo did not report hardware NVIDIA");
      const pdb = await page.evaluate(
        async () =>
          await (await fetch(document.getElementById("download").href)).text(),
      );
      await page.screenshot({ path: root + ".cache/demo.png" });
      return { text, pdb };
    });
    assert.equal(
      result.pdb.split("\n").filter((l) => l.startsWith("ATOM")).length,
      85,
    );
    assert.match(result.text, /Generated 17 residues/);
  },
);
