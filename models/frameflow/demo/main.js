import { createFrameFlowWorker } from "../webgpu/worker.js";
const $ = (id) => document.getElementById(id),
  status = $("status"),
  canvas = $("view"),
  ctx = canvas.getContext("2d");
let engine,
  loadedPrecision,
  controller,
  coords,
  url,
  angle = 0;
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!coords) return;
  const n = coords.length / 3,
    center = [0, 0, 0];
  for (let i = 0; i < coords.length; i++) center[i % 3] += coords[i] / n;
  const points = [];
  for (let i = 0; i < n; i++) {
    const x = coords[i * 3] - center[0],
      y = coords[i * 3 + 1] - center[1],
      z = coords[i * 3 + 2] - center[2];
    points.push([
      x * Math.cos(angle) + z * Math.sin(angle),
      y,
      -x * Math.sin(angle) + z * Math.cos(angle),
    ]);
  }
  const max = Math.max(
      1,
      ...points.flatMap((p) => [Math.abs(p[0]), Math.abs(p[1])]),
    ),
    scale = 190 / max;
  ctx.lineWidth = 3;
  for (let i = 1; i < n; i++) {
    ctx.strokeStyle = `hsl(${145 + (100 * i) / n} 65% 65%)`;
    ctx.beginPath();
    ctx.moveTo(500 + points[i - 1][0] * scale, 230 - points[i - 1][1] * scale);
    ctx.lineTo(500 + points[i][0] * scale, 230 - points[i][1] * scale);
    ctx.stroke();
  }
}
canvas.onpointerdown = (e) => {
  canvas.setPointerCapture(e.pointerId);
};
canvas.onpointermove = (e) => {
  if (e.buttons) {
    angle += e.movementX * 0.01;
    draw();
  }
};
$("cancel").onclick = () => controller?.abort();
$("controls").onsubmit = async (e) => {
  e.preventDefault();
  controller = new AbortController();
  $("run").disabled = true;
  $("cancel").disabled = false;
  $("download").hidden = true;
  const precision = $("precision").value;
  try {
    if (!engine || loadedPrecision !== precision) {
      await engine?.destroy();
      engine = null;
      status.textContent = "Loading weights…";
      engine = await createFrameFlowWorker({
        baseUrl: new URL("../weights", import.meta.url).href,
        precision,
        cache: "cache",
        signal: controller.signal,
        onProgress: (p) => {
          if (p.phase === "weights")
            status.textContent = `Loading weights: ${Math.round((100 * p.loaded) / p.total)}%`;
        },
      });
      loadedPrecision = precision;
    }
    const start = performance.now();
    const r = await engine.sample({
      length: Number($("length").value),
      steps: Number($("steps").value),
      signal: controller.signal,
      onProgress: (p) => {
        if (p.phase === "step")
          status.textContent = `Generating: step ${p.step} of ${p.total}`;
      },
    });
    coords = r.pred_trans;
    draw();
    if (url) URL.revokeObjectURL(url);
    url = URL.createObjectURL(new Blob([r.pdb], { type: "chemical/x-pdb" }));
    $("download").href = url;
    $("download").hidden = false;
    status.textContent = `Generated ${coords.length / 3} residues in ${((performance.now() - start) / 1000).toFixed(1)} seconds. ${engine.adapterInfo.description || engine.adapterInfo.vendor}${engine.isFallbackAdapter ? " (software adapter)" : ""}.`;
  } catch (e) {
    status.textContent =
      e.name === "AbortError" ? "Generation cancelled." : e.message;
  } finally {
    $("run").disabled = false;
    $("cancel").disabled = true;
    controller = null;
  }
};
window.addEventListener("pagehide", () => {
  engine?.destroy();
  if (url) URL.revokeObjectURL(url);
});
