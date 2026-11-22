const $ = (id) => document.getElementById(id);
let worker,
  currentPrecision,
  serial = 0,
  result,
  coords,
  angle = 0.5,
  tilt = 0.35;
const pending = new Map();
function setStatus(text) {
  $("status").textContent = text;
}
function startWorker() {
  worker = new Worker("../webgpu/worker.js", { type: "module" });
  worker.onmessage = ({ data }) => {
    if (data.progress) {
      const p = data.progress;
      $("progress").value =
        p.phase === "step"
          ? p.fraction
          : p.phase === "weights"
            ? p.loaded / p.total
            : ((p.step ?? 1) - 1 + p.block / p.total) / $("steps").value;
      setStatus(
        p.phase === "weights"
          ? `Loading weights: ${Math.round((p.loaded / p.total) * 100)}%`
          : `Step ${p.step ?? 1}/${p.numSteps ?? $("steps").value}, block ${p.block ?? "complete"}`,
      );
      return;
    }
    const task = pending.get(data.id);
    if (!task) return;
    pending.delete(data.id);
    data.error
      ? task.reject(
          Object.assign(Error(data.error.message), { name: data.error.name }),
        )
      : task.resolve(data.result);
  };
  worker.onerror = (e) => {
    for (const p of pending.values()) p.reject(Error(e.message));
    pending.clear();
  };
}
const call = (method, args = {}) =>
  new Promise((resolve, reject) => {
    pending.set(++serial, { resolve, reject });
    worker.postMessage({ id: serial, method, ...args });
  });
$("mode").onchange = () => {
  $("motifFields").hidden = $("mode").value !== "motif";
  $("symmetryFields").hidden = $("mode").value !== "symmetric";
};
$("form").onsubmit = async (e) => {
  e.preventDefault();
  $("run").disabled = true;
  $("cancel").disabled = false;
  try {
    if (!worker) startWorker();
    const precision = $("precision").value;
    if (currentPrecision !== precision) {
      currentPrecision = null;
      await call("create", {
        options: {
          baseUrl: new URL("../weights", location.href).href,
          precision,
          cache: "cache",
        },
      });
      currentPrecision = precision;
    }
    const options = {
      length: Number($("length").value),
      numSteps: Number($("steps").value),
      seed: Number($("seed").value),
    };
    if ($("mode").value === "symmetric") options.symmetry = $("symmetry").value;
    if ($("mode").value === "motif") {
      const file = $("pdb").files[0];
      if (!file) throw Error("Choose a motif PDB");
      options.inputPdb = await file.text();
      options.contig = $("contig").value;
    }
    result = await call("sample", { options });
    coords = result.coords;
    $("frame").max = result.trajectory.length - 1;
    $("frame").value = result.trajectory.length - 1;
    $("frame").disabled = false;
    $("download").disabled = false;
    draw();
    setStatus(
      `Finished ${result.trajectory.length} steps in ${(result.timingMs / 1000).toFixed(1)} s`,
    );
  } catch (error) {
    setStatus(error.name === "AbortError" ? "Cancelled." : error.message);
  } finally {
    $("run").disabled = false;
    $("cancel").disabled = true;
  }
};
$("cancel").onclick = () => worker?.postMessage({ method: "abort" });
$("release").onclick = async () => {
  if (worker) {
    worker.postMessage({ method: "abort" });
    try {
      await call("destroy");
    } catch {
    } finally {
      worker.terminate();
      worker = null;
      currentPrecision = null;
      for (const p of pending.values()) p.reject(Error("GPU released"));
      pending.clear();
    }
  }
  setStatus("GPU released.");
};
$("frame").oninput = () => {
  coords = result.trajectory[Number($("frame").value)].px0;
  draw();
};
$("download").onclick = () => {
  const url = URL.createObjectURL(
      new Blob([result.pdbText], { type: "chemical/x-pdb" }),
    ),
    a = document.createElement("a");
  a.href = url;
  a.download = "rfdiffusion.pdb";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
function draw() {
  const canvas = $("view"),
    ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!coords) return;
  const L = result.sequence.length,
    n = coords.length / L / 3,
    points = Array.from({ length: L }, (_, i) =>
      Array.from(coords.slice((i * n + 1) * 3, (i * n + 2) * 3)),
    ),
    center = [0, 1, 2].map((d) => points.reduce((s, p) => s + p[d], 0) / L),
    radius = Math.max(
      1,
      ...points.map((p) => Math.hypot(...p.map((x, j) => x - center[j]))),
    ),
    scale = 260 / radius;
  ctx.lineWidth = 4;
  ctx.strokeStyle = "#90dac8";
  ctx.beginPath();
  points.forEach((p, i) => {
    const [x, y, z] = p.map((v, j) => v - center[j]),
      rx = x * Math.cos(angle) + z * Math.sin(angle),
      rz = z * Math.cos(angle) - x * Math.sin(angle),
      ry = y * Math.cos(tilt) - rz * Math.sin(tilt),
      px = canvas.width / 2 + rx * scale,
      py = canvas.height / 2 - ry * scale;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();
}
let drag;
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
$("view").onpointerup = () => (drag = null);
