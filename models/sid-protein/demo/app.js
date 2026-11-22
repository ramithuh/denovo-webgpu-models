import { createSidProteinWorker } from "../webgpu/worker-client.js";
const $ = (id) => document.getElementById(id),
  ctx = $("view").getContext("2d");
let client, loaded, controller, downloadURL;
function draw(coords) {
  const w = $("view").width,
    h = $("view").height;
  ctx.clearRect(0, 0, w, h);
  if (!coords) return;
  const points = [];
  for (let i = 0; i < coords.length; i += 3)
    points.push([
      coords[i] + 0.35 * coords[i + 2],
      coords[i + 1] - 0.2 * coords[i + 2],
    ]);
  const xs = points.map((p) => p[0]),
    ys = points.map((p) => p[1]),
    minX = Math.min(...xs),
    maxX = Math.max(...xs),
    minY = Math.min(...ys),
    maxY = Math.max(...ys),
    scale =
      0.8 *
      Math.min(w / Math.max(maxX - minX, 0.1), h / Math.max(maxY - minY, 0.1));
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  for (let i = 1; i < points.length; i++) {
    ctx.strokeStyle = `hsl(${155 + (100 * i) / points.length} 65% 70%)`;
    ctx.beginPath();
    ctx.moveTo(
      w / 2 + (points[i - 1][0] - (minX + maxX) / 2) * scale,
      h / 2 + (points[i - 1][1] - (minY + maxY) / 2) * scale,
    );
    ctx.lineTo(
      w / 2 + (points[i][0] - (minX + maxX) / 2) * scale,
      h / 2 + (points[i][1] - (minY + maxY) / 2) * scale,
    );
    ctx.stroke();
  }
}
async function release() {
  if (client) {
    await client.destroy();
    client = null;
    loaded = null;
  }
  $("status").textContent = "GPU released.";
}
$("cancel").onclick = () => controller?.abort();
$("release").onclick = () =>
  release().catch((e) => ($("status").textContent = e.message));
$("form").onsubmit = async (event) => {
  event.preventDefault();
  controller = new AbortController();
  $("generate").disabled = true;
  $("cancel").disabled = false;
  $("release").disabled = true;
  try {
    const precision = $("precision").value;
    if (client && loaded !== precision) await release();
    if (!client) {
      client = createSidProteinWorker();
      const meta = await client.call(
        "init",
        {
          baseUrl: new URL(`../weights/${precision}`, import.meta.url).href,
          precision,
          cache: "cache",
        },
        {
          signal: controller.signal,
          onProgress: (_, v) => {
            $("status").textContent =
              v.phase === "weights"
                ? `Loading weights: ${Math.round((v.loaded / v.total) * 100)}%`
                : (v.message ?? v.phase);
          },
        },
      );
      loaded = precision;
      $("status").textContent =
        `Ready on ${meta.adapterInfo.vendor || "WebGPU"}${meta.isFallbackAdapter ? " (software adapter)" : ""}`;
    }
    const start = performance.now(),
      result = await client.call(
        "generate",
        {
          length: Number($("length").value),
          steps: Number($("steps").value),
          noiseScale: Number($("noise").value),
          seed: Number($("seed").value),
        },
        {
          signal: controller.signal,
          onProgress: (kind, v) => {
            if (kind === "step") {
              draw(v.coordinates);
              $("status").textContent =
                `Generating: step ${v.step} of ${v.steps}`;
            }
          },
        },
      );
    draw(result.coordinates);
    $("status").textContent =
      `${result.coordinates.length / 3} residues · ${((performance.now() - start) / 1000).toFixed(2)} seconds`;
    if (downloadURL) URL.revokeObjectURL(downloadURL);
    downloadURL = URL.createObjectURL(
      new Blob([result.pdb], { type: "chemical/x-pdb" }),
    );
    $("download").href = downloadURL;
    $("download").hidden = false;
  } catch (e) {
    $("status").textContent =
      e.name === "AbortError" ? "Generation cancelled." : e.message;
    if (!loaded) {
      await client?.destroy();
      client = null;
    }
  } finally {
    $("generate").disabled = false;
    $("cancel").disabled = true;
    $("release").disabled = false;
  }
};
