import test from "node:test";
import assert from "node:assert/strict";
import { browserTask } from "./browser.mjs";
test(
  "static demo generates a PDB in its module worker and releases GPU",
  { timeout: 120000 },
  async () => {
    await browserTask(async (page, url) => {
      await page.goto(url + "/demo/index.html");
      await page.locator("#steps").fill("2");
      await page.locator("#run").click();
      await page.waitForFunction(
        () =>
          document.querySelector("#status").textContent.startsWith("Finished"),
        {},
        { timeout: 90000 },
      );
      assert.equal(await page.locator("#download").isEnabled(), true);
      const download = page.waitForEvent("download");
      await page.locator("#download").click();
      assert.equal((await download).suggestedFilename(), "rfdiffusion.pdb");
      await page.locator("#release").click();
      await page.waitForFunction(
        () => document.querySelector("#status").textContent === "GPU released.",
      );
    });
  },
);
