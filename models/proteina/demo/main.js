import { createProteinaWorker } from "../webgpu/worker-client.js";
import { toPDB } from "../shared/index.js";
const $ = (id) => document.getElementById(id),
  canvas = $("view"),
  ctx = canvas.getContext("2d");
let worker,
  loadedPrecision,
  controller,
  coordinates,
  angle = 0.6,
  tilt = 0.4,
  drag;
function draw() {
  ctx.clearRect(0, 0, 720, 720);
  if (!coordinates) return;
  const pts = [];
  let extent = 1;
  for (let i = 0; i < coordinates.length; i += 3) {
    const x = coordinates[i],
      y = coordinates[i + 1],
      z = coordinates[i + 2],
      rx = x * Math.cos(angle) + z * Math.sin(angle),
      rz = z * Math.cos(angle) - x * Math.sin(angle),
      ry = y * Math.cos(tilt) - rz * Math.sin(tilt);
    pts.push([rx, ry]);
    extent = Math.max(extent, Math.abs(rx), Math.abs(ry));
  }
  const scale = 300 / extent;
  ctx.lineWidth = 3;
  for (let i = 1; i < pts.length; i++) {
    ctx.strokeStyle = `hsl(${145 + (120 * i) / pts.length} 65% 65%)`;
    ctx.beginPath();
    ctx.moveTo(360 + pts[i - 1][0] * scale, 360 + pts[i - 1][1] * scale);
    ctx.lineTo(360 + pts[i][0] * scale, 360 + pts[i][1] * scale);
    ctx.stroke();
  }
}
canvas.onpointerdown = (e) => {
  drag = [e.clientX, e.clientY];
  canvas.setPointerCapture(e.pointerId);
};
canvas.onpointerup = () => (drag = null);
canvas.onpointermove = (e) => {
  if (!drag) return;
  angle += (e.clientX - drag[0]) * 0.01;
  tilt += (e.clientY - drag[1]) * 0.01;
  drag = [e.clientX, e.clientY];
  draw();
};
function busy(value) {
  $("run").disabled = value;
  $("cancel").disabled = !value;
  $("release").disabled = value;
}
$("form").onsubmit = async (event) => {
  event.preventDefault();
  controller = new AbortController();
  busy(true);
  $("download").disabled = true;
  try {
    const precision = $("precision").value;
    if (!worker || loadedPrecision !== precision) {
      await worker?.destroy();
      worker = createProteinaWorker();
      loadedPrecision = null;
      $("status").textContent = "Loading weights…";
      const info = await worker.init(
        {
          baseUrl: new URL("../weights", import.meta.url).href,
          precision,
          cache: "cache",
        },
        {
          signal: controller.signal,
          onProgress: (p) => {
            if (p.phase === "weights")
              $("status").textContent =
                `Loading weights: ${Math.round((100 * p.loaded) / p.total)}%`;
          },
        },
      );
      loadedPrecision = precision;
      $("adapter").textContent =
        `Adapter: ${info.adapterInfo.vendor || info.adapterInfo.description || "WebGPU"}${info.isFallbackAdapter ? " (software fallback)" : ""}`;
    }
    const foldText = $("fold").value.trim(),
      fold = foldText ? foldText.split(",").map(Number) : undefined;
    $("status").textContent = "Generating…";
    const started = performance.now();
    const result = await worker.sample(
      {
        length: Number($("length").value),
        steps: Number($("steps").value),
        seed: Number($("seed").value),
        mode: $("mode").value,
        schedule: $("schedule").value,
        noiseScale: Number($("noise").value),
        selfCondition: $("self-condition").checked,
        fold,
      },
      {
        signal: controller.signal,
        onProgress: (p) => {
          if (p.phase === "sampling") {
            coordinates = p.coordinates;
            draw();
            $("status").textContent = `Generating: ${p.step} / ${p.steps}`;
          }
        },
      },
    );
    coordinates = result.coordinates;
    draw();
    $("status").textContent =
      `Done. ${result.length} residues in ${((performance.now() - started) / 1000).toFixed(2)} s.`;
    $("download").disabled = false;
  } catch (e) {
    $("status").textContent =
      e.name === "AbortError" ? "Cancelled." : "Error: " + e.message;
  } finally {
    busy(false);
    controller = null;
  }
};
$("cancel").onclick = () => controller?.abort();
$("release").onclick = async () => {
  await worker?.destroy();
  worker = null;
  loadedPrecision = null;
  $("status").textContent = "GPU buffers released.";
};
$("download").onclick = () => {
  const blob = new Blob([toPDB(coordinates)], { type: "chemical/x-pdb" }),
    url = URL.createObjectURL(blob),
    a = document.createElement("a");
  a.href = url;
  a.download = "proteina-ca.pdb";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
