import test from "node:test";
import assert from "node:assert/strict";
import { browserTask, root } from "./browser.mjs";
test(
  "standalone demo generates coordinates, downloads, rotates and releases GPU",
  { timeout: 180000 },
  async () => {
    const result = await browserTask(async (page, url) => {
      await page.setViewportSize({ width: 1200, height: 850 });
      await page.goto(url + "/demo/");
      await page.selectOption("#mode", "binder");
      await page.selectOption("#precision", "int8");
      await page.fill("#length", "8");
      await page.fill("#numSteps", "5");
      await page.fill("#seed", "46");
      await page.click("#run");
      await page.waitForFunction(
        () =>
          document.querySelector("#status").textContent.startsWith("Generated"),
        null,
        { timeout: 120000 },
      );
      const text = await page.locator("#status").textContent();
      await page.locator("#view").hover();
      await page.mouse.down();
      await page.mouse.move(800, 390, { steps: 8 });
      await page.mouse.up();
      await page.screenshot({ path: root + "/.cache/demo.png" });
      const downloadPromise = page.waitForEvent("download");
      await page.click("#download");
      const download = await downloadPromise;
      await page.click("#release");
      return {
        text,
        download: download.suggestedFilename(),
        released: await page.locator("#status").textContent(),
      };
    });
    assert.match(result.text, /Generated 160 atom slots/);
    assert.equal(result.download, "boltzgen-coordinates.json");
    assert.equal(result.released, "GPU resources released.");
  },
);
