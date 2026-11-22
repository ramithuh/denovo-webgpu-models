import { createBoltzGenEngine } from "../webgpu/index.js";
const $ = (id) => document.getElementById(id);
let engine,
  key,
  controller,
  result,
  angle = 0.3,
  tilt = 0.2,
  drag;
$("mode").onchange = () => {
  $("targetLabel").hidden = $("mode").value !== "binder";
};
function draw() {
  const canvas = $("view"),
    ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!result) return;
  const coords = result.atom_coords,
    center = [0, 0, 0];
  for (let i = 0; i < coords.length; i++)
    center[i % 3] += coords[i] / result.nAtoms;
  const points = [];
  let extent = 1;
  for (let i = 0; i < result.nAtoms; i++) {
    const x = coords[i * 3] - center[0],
      y = coords[i * 3 + 1] - center[1],
      z = coords[i * 3 + 2] - center[2],
      rx = x * Math.cos(angle) + z * Math.sin(angle),
      rz = -x * Math.sin(angle) + z * Math.cos(angle),
      ry = y * Math.cos(tilt) - rz * Math.sin(tilt);
    points.push({
      x: rx,
      y: ry,
      z: y * Math.sin(tilt) + rz * Math.cos(tilt),
      i,
    });
    extent = Math.max(extent, Math.hypot(rx, ry));
  }
  const scale = (Math.min(canvas.width, canvas.height) * 0.4) / extent;
  points.sort((a, b) => a.z - b.z);
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(
      canvas.width / 2 + p.x * scale,
      canvas.height / 2 - p.y * scale,
      4.5,
      0,
      2 * Math.PI,
    );
    ctx.fillStyle = !result.atoms[p.i].designed ? "#79b5ef" : "#79d9c9";
    ctx.fill();
  }
}
$("view").onpointerdown = (e) => {
  drag = [e.clientX, e.clientY];
  $("view").setPointerCapture(e.pointerId);
};
$("view").onpointermove = (e) => {
  if (drag) {
    angle += (e.clientX - drag[0]) * 0.01;
    tilt += (e.clientY - drag[1]) * 0.01;
    drag = [e.clientX, e.clientY];
    draw();
  }
};
$("view").onpointerup = () => {
  drag = null;
};
$("run").onclick = async () => {
  controller = new AbortController();
  $("run").disabled = true;
  $("cancel").disabled = false;
  $("release").disabled = true;
  $("download").disabled = true;
  try {
    const nextKey = $("variant").value + "/" + $("precision").value;
    if (nextKey !== key) {
      engine?.destroy();
      engine = null;
      key = null;
    }
    if (!engine) {
      $("status").textContent = "Loading and verifying weight shards…";
      engine = await createBoltzGenEngine({
        baseUrl: new URL("../weights/" + nextKey + "/", import.meta.url).href,
        precision: $("precision").value,
        signal: controller.signal,
        onProgress: ({ loaded, total }) => {
          $("progress").value = loaded / total;
        },
      });
      key = nextKey;
    }
    $("status").textContent =
      `Running the trunk and ${$("numSteps").value} all-atom diffusion steps…`;
    const start = performance.now();
    result = await engine.generate({
      mode: $("mode").value,
      seed: Number($("seed").value),
      L: Number($("length").value),
      numSteps: Number($("numSteps").value),
      recyclingSteps: Number($("recyclingSteps").value),
      targetPDB: $("mode").value === "binder" ? $("target").value : undefined,
      signal: controller.signal,
      onProgress: ({ phase, block, step, total }) => {
        $("progress").value =
          phase === "trunk"
            ? (0.4 * block) / total
            : 0.4 + (0.6 * step) / total;
      },
    });
    $("progress").value = 1;
    $("status").textContent =
      `Generated ${result.nAtoms} atom slots in ${((performance.now() - start) / 1000).toFixed(2)} seconds.`;
    $("metrics").textContent = result.timings
      .map((t) => `${t.phase}: ${t.milliseconds.toFixed(1)} ms`)
      .join("\n");
    $("download").disabled = false;
    draw();
  } catch (error) {
    $("status").textContent =
      error.name === "AbortError" ? "Cancelled." : error.message;
  } finally {
    $("run").disabled = false;
    $("cancel").disabled = true;
    $("release").disabled = !engine;
  }
};
$("cancel").onclick = () => controller?.abort();
$("release").onclick = () => {
  engine?.destroy();
  engine = null;
  key = null;
  $("release").disabled = true;
  $("status").textContent = "GPU resources released.";
};
$("download").onclick = () => {
  const value = {
      nAtoms: result.nAtoms,
      atom_coords: Array.from(result.atom_coords),
      denoised: result.denoised.map((x) => Array.from(x)),
      trajectory: result.trajectory.map((x) => Array.from(x)),
    },
    url = URL.createObjectURL(
      new Blob([JSON.stringify(value)], { type: "application/json" }),
    ),
    link = document.createElement("a");
  link.href = url;
  link.download = "boltzgen-coordinates.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
addEventListener("pagehide", () => engine?.destroy());

fetch(new URL("../fixtures/1crn-12.pdb", import.meta.url))
  .then((r) => r.text())
  .then((text) => {
    $("target").value = text;
  })
  .catch(() => {});
