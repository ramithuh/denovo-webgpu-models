import { parseStructure, toPDB } from "../shared/structure.js";
import { constants } from "../shared/constants.js";
const $ = (id) => document.getElementById(id),
  worker = new Worker("../webgpu/worker.js", { type: "module" }),
  pending = new Map();
let id = 0,
  loaded = "",
  result,
  indices,
  chains;
worker.onmessage = ({ data }) => {
  if (data.progress) {
    $("progress").value =
      data.progress.loaded / Math.max(1, data.progress.total);
    $("status").textContent =
      data.progress.phase === "weights"
        ? "Loading model weights…"
        : `Sampling step ${data.progress.loaded} of ${data.progress.total}`;
    return;
  }
  const p = pending.get(data.id);
  if (!p) return;
  pending.delete(data.id);
  data.error
    ? p.reject(
        Object.assign(new Error(data.error.message), { name: data.error.name }),
      )
    : p.resolve(data.result);
};
worker.onerror = (e) => {
  for (const p of pending.values()) p.reject(new Error(e.message));
  pending.clear();
};
const rpc = (method, options = {}) =>
  new Promise((resolve, reject) => {
    const request = ++id;
    pending.set(request, { resolve, reject });
    worker.postMessage({ id: request, method, ...options });
  });
const busy = (value) => {
  $("generate").disabled = value;
  $("cancel").disabled = !value;
  $("release").disabled = value || !loaded;
};
$("mode").onchange = () => {
  $("fileControls").hidden = !["motif", "partial"].includes($("mode").value);
};
$("cancel").onclick = () => worker.postMessage({ method: "cancel" });
$("release").onclick = async () => {
  await rpc("destroy");
  loaded = "";
  $("release").disabled = true;
  $("memory").textContent = "0 MB";
  $("status").textContent = "GPU resources released.";
};
$("generate").onclick = async () => {
  busy(true);
  $("status").textContent = "Preparing model…";
  $("progress").value = 0;
  try {
    const mode = $("mode").value,
      precision = $("precision").value,
      variant =
        mode === "allatom"
          ? "cc91_epoch383"
          : mode === "multichain"
            ? "cc78_epoch1431"
            : "cc58_epoch416";
    let n = Number($("length").value),
      structure;
    const steps = Number($("steps").value);
    if (
      !Number.isInteger(n) ||
      n < 4 ||
      n > 512 ||
      !Number.isInteger(steps) ||
      steps < 2 ||
      steps > 1000
    )
      throw new Error("Enter 4–512 residues and 2–1000 steps.");
    if (["motif", "partial"].includes(mode)) {
      const file = $("structure").files[0];
      if (!file) throw new Error("Select an input structure.");
      structure = parseStructure(await file.text(), {
        format: file.name.endsWith(".pdb") ? "pdb" : "cif",
      });
      if (mode === "partial") n = structure.aatype.length;
      if (structure.aatype.length < 3)
        throw new Error("The motif needs at least three residues.");
    }
    const key = variant + "/" + precision;
    if (key !== loaded) {
      await rpc("load", {
        options: {
          baseUrl: new URL("../weights", location.href).href,
          variant,
          precision,
          cache: "cache",
        },
      });
      loaded = key;
    }
    const count = mode === "multichain" ? n * 2 : n;
    indices = Float32Array.from({ length: count }, (_, i) =>
      mode === "multichain" ? (i % n) + 1 : i + 1,
    );
    chains = Int32Array.from({ length: count }, (_, i) => Math.floor(i / n));
    const input = {
      seqMask: new Float32Array(count).fill(1),
      residueIndex: indices,
      chainIndex: chains,
      numSteps: steps,
      sidechainMode: mode === "allatom",
      jumpSteps: mode === "allatom",
      skipMPNNProportion: 0.6,
      annealSequence: "linear",
    };
    if (mode === "motif")
      input.motif = {
        structure,
        sourceIndices: [0, 1, 2],
        targetIndices: [2, 3, 4],
        strategy: $("crop").value,
      };
    if (mode === "partial") {
      input.partial = {
        structure,
        rewindSteps: Math.max(1, Math.round(steps / 4)),
      };
      indices = structure.originalIndex;
      chains = structure.chainIndex;
      input.residueIndex = structure.residueIndex;
      input.chainIndex = chains;
    }
    result = await rpc("sample", { input });
    result.residueIndex = indices;
    result.chainIndex = chains;
    $("residues").textContent = result.aatype.length;
    $("timing").textContent = (result.timingMs / 1000).toFixed(2) + " s";
    $("memory").textContent = (result.memory.bytes / 1e6).toFixed(1) + " MB";
    $("sequence").textContent = sequence();
    $("pdb").disabled = $("fasta").disabled = false;
    $("status").textContent = "Generation complete.";
    $("progress").value = 1;
    draw();
  } catch (e) {
    $("status").textContent =
      e.name === "AbortError" ? "Generation cancelled." : e.message;
  } finally {
    busy(false);
  }
};
function sequence() {
  return Array.from(result.aatype, (i) => constants.restypes[i] ?? "X").join(
    "",
  );
}
function download(name, text) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain" })),
    a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("pdb").onclick = () => download("protpardelle.pdb", toPDB(result));
$("fasta").onclick = () =>
  download("protpardelle.fasta", ">protpardelle\n" + sequence() + "\n");
let yaw = 0.5,
  pitch = 0.3,
  drag;
const canvas = $("view");
canvas.onpointerdown = (e) => {
  drag = [e.clientX, e.clientY];
  canvas.setPointerCapture(e.pointerId);
};
canvas.onpointerup = () => (drag = null);
canvas.onpointermove = (e) => {
  if (!drag) return;
  yaw += (e.clientX - drag[0]) * 0.01;
  pitch += (e.clientY - drag[1]) * 0.01;
  drag = [e.clientX, e.clientY];
  draw();
};
function draw() {
  const dpr = devicePixelRatio || 1,
    rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  if (!result) {
    ctx.fillStyle = "#8196ae";
    ctx.textAlign = "center";
    ctx.font = "16px system-ui";
    ctx.fillText(
      "Generate a protein to explore its structure",
      rect.width / 2,
      rect.height / 2,
    );
    return;
  }
  const points = Array.from(result.aatype, (_, i) =>
      Array.from(result.coords.slice(i * 111 + 3, i * 111 + 6)),
    ),
    mean = [0, 1, 2].map(
      (c) => points.reduce((s, p) => s + p[c], 0) / points.length,
    ),
    rotated = points.map((p) => {
      const [x, y, z] = p.map((v, c) => v - mean[c]),
        u = x * Math.cos(yaw) + z * Math.sin(yaw),
        v = z * Math.cos(yaw) - x * Math.sin(yaw);
      return [
        u,
        y * Math.cos(pitch) - v * Math.sin(pitch),
        y * Math.sin(pitch) + v * Math.cos(pitch),
      ];
    }),
    radius = Math.max(1, ...rotated.map((p) => Math.hypot(...p))),
    scale = (Math.min(rect.width, rect.height) * 0.4) / radius;
  rotated.forEach((p, i) => {
    if (i && chains[i] === chains[i - 1]) {
      ctx.strokeStyle = chains[i] % 2 ? "#d4adf3" : "#85d4bf";
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      ctx.moveTo(
        rect.width / 2 + rotated[i - 1][0] * scale,
        rect.height / 2 + rotated[i - 1][1] * scale,
      );
      ctx.lineTo(rect.width / 2 + p[0] * scale, rect.height / 2 + p[1] * scale);
      ctx.stroke();
    }
    ctx.fillStyle = chains[i] % 2 ? "#e0c3f5" : "#abe4d4";
    ctx.beginPath();
    ctx.arc(
      rect.width / 2 + p[0] * scale,
      rect.height / 2 + p[1] * scale,
      3,
      0,
      Math.PI * 2,
    );
    ctx.fill();
  });
}
new ResizeObserver(draw).observe(canvas);
