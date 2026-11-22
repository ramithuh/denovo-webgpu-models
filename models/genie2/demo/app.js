import { createGenie2Worker } from "../webgpu/worker-client.js";
import { parseMotifSpecification, writePDB } from "../shared/index.js";
const $ = (id) => document.getElementById(id),
  canvas = $("view"),
  ctx = canvas.getContext("2d");
let worker,
  precision,
  controller,
  result,
  coords,
  yaw = 0.4,
  pitch = 0.2,
  last;
const status = (text) => ($("status").textContent = text);
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!coords) return;
  const L = coords.length / 3,
    center = [0, 0, 0];
  for (let i = 0; i < L; i++)
    for (let k = 0; k < 3; k++) center[k] += coords[3 * i + k] / L;
  let radius = 1;
  const points = [];
  for (let i = 0; i < L; i++) {
    const x = coords[i * 3] - center[0],
      y = coords[i * 3 + 1] - center[1],
      z = coords[i * 3 + 2] - center[2],
      a = x * Math.cos(yaw) + z * Math.sin(yaw),
      b = z * Math.cos(yaw) - x * Math.sin(yaw),
      c = y * Math.cos(pitch) - b * Math.sin(pitch);
    points.push([a, c]);
    radius = Math.max(radius, Math.abs(a), Math.abs(c));
  }
  const scale = (Math.min(canvas.width, canvas.height) * 0.42) / radius;
  ctx.lineWidth = 4;
  for (let i = 1; i < L; i++) {
    ctx.strokeStyle = `hsl(${160 + (i / L) * 160} 70% 65%)`;
    ctx.beginPath();
    ctx.moveTo(
      canvas.width / 2 + points[i - 1][0] * scale,
      canvas.height / 2 + points[i - 1][1] * scale,
    );
    ctx.lineTo(
      canvas.width / 2 + points[i][0] * scale,
      canvas.height / 2 + points[i][1] * scale,
    );
    ctx.stroke();
  }
}
canvas.onpointerdown = (e) => {
  last = [e.clientX, e.clientY];
  canvas.setPointerCapture(e.pointerId);
};
canvas.onpointermove = (e) => {
  if (!last) return;
  yaw += (e.clientX - last[0]) * 0.01;
  pitch += (e.clientY - last[1]) * 0.01;
  last = [e.clientX, e.clientY];
  draw();
};
canvas.onpointerup = () => (last = null);
$("generate").onclick = async () => {
  $("generate").disabled = true;
  $("cancel").disabled = false;
  $("save").disabled = true;
  controller = new AbortController();
  try {
    const p = $("precision").value;
    if (!worker || p !== precision) {
      await worker?.destroy();
      worker = createGenie2Worker();
      status("Loading weights…");
      const info = await worker.init({
        baseUrl: new URL(`../weights/${p}`, location.href).href,
        precision: p,
        cache: "opfs",
      });
      precision = p;
      status(
        `Adapter: ${info.adapterInfo.vendor || "available"}${info.isFallbackAdapter ? " (software fallback)" : ""}`,
      );
    }
    const steps = Number($("steps").value),
      options = {
        length: Number($("length").value),
        steps,
        scale: Number($("scale").value),
        seed: Number($("seed").value),
      };
    if ($("motif").value.trim())
      options.scaffold = parseMotifSpecification($("motif").value);
    result = await worker.sample(options, {
      signal: controller.signal,
      onStep: ({ step, coordinates }) => {
        coords = coordinates;
        draw();
        status(`Completed ${steps - step + 1} / ${steps} steps`);
      },
    });
    coords = result.coordinates;
    draw();
    $("save").disabled = false;
    status(`Generated ${coords.length / 3} residues.`);
  } catch (e) {
    status(e.name === "AbortError" ? "Cancelled." : e.message);
  } finally {
    $("generate").disabled = false;
    $("cancel").disabled = true;
  }
};
$("cancel").onclick = () => controller?.abort();
$("release").onclick = async () => {
  controller?.abort();
  await worker?.destroy();
  worker = null;
  precision = null;
  status("GPU released.");
};
$("save").onclick = () => {
  const url = URL.createObjectURL(
      new Blob([writePDB(result)], { type: "chemical/x-pdb" }),
    ),
    a = document.createElement("a");
  a.href = url;
  a.download = "genie2-ca.pdb";
  a.click();
  URL.revokeObjectURL(url);
};
