import { bestMutants, toCSV, toFASTA } from "../shared/results.js";
const $ = (id) => document.getElementById(id);
let worker,
  sequence = 0,
  pending = new Map(),
  loadedKey = "",
  result,
  sourceText = "",
  sourceName = "structure.pdb",
  busy = false;
function startWorker() {
  worker = new Worker(new URL("../webgpu/worker.js", import.meta.url), {
    type: "module",
  });
  worker.onmessage = ({ data }) => {
    if (data.ready) return;
    const p = pending.get(data.id);
    if (!p) return;
    if (data.progress) {
      const phase = data.progress.phase;
      $("status").textContent =
        phase === "weights"
          ? "Loading verified model weights…"
          : `Computing ${phase.replace("_", " ")}…`;
      $("progress").value =
        phase === "weights"
          ? (25 * data.progress.loaded) / data.progress.total
          : 30 +
            10 * (Number(phase.at(-1)) + (phase.startsWith("decoder") ? 3 : 0));
      return;
    }
    pending.delete(data.id);
    if (data.error) p.reject(new Error(data.error));
    else p.resolve(data.result);
  };
  worker.onerror = (e) => {
    for (const p of pending.values()) p.reject(new Error(e.message));
    pending.clear();
  };
}
function call(command, options, text) {
  const id = ++sequence;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, command, options, text });
  });
}
function setBusy(value) {
  busy = value;
  for (const id of [
    "run",
    "release",
    "file",
    "example",
    "precision",
    "cache",
    "base",
    "mode",
  ])
    $(id).disabled = value;
  $("cancel").disabled = !value;
}
function mutations() {
  return $("mutations")
    .value.trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => {
      const m = /^(.+):(-?\d+)([A-Za-z]?):([ACDEFGHIKLMNPQRSTVWY])$/.exec(
        line.trim(),
      );
      if (!m)
        throw new Error(
          `Invalid mutation: ${line}. Use chain:number[insertion code]:mutant.`,
        );
      return { chain: m[1], number: Number(m[2]), ins: m[3], mutation: m[4] };
    });
}
function renderTable() {
  $("rows").replaceChildren();
  if (!result) return;
  const rows = result.mutations
    ? result.mutations.map((m, i) => ({ ...m, ddG: result.scores[i] }))
    : bestMutants(result, { includeCys: $("cys").checked });
  $("table-title").textContent = result.mutations
    ? "Targeted point mutants"
    : "Best substitutions per site";
  for (const r of rows) {
    const tr = document.createElement("tr");
    for (const v of [
      r.chain,
      `${r.number}${r.ins}${r.structuralValid ? "" : " †"}`,
      r.aa,
      r.mutation,
      r.ddG.toFixed(3),
    ]) {
      const td = document.createElement("td");
      td.textContent = v;
      tr.append(td);
    }
    $("rows").append(tr);
  }
}
const cell = 18,
  pad = 28,
  top = 12;
function render() {
  $("empty").hidden = true;
  const canvas = $("heatmap");
  canvas.hidden = false;
  canvas.width = Math.max(500, result.shape[0] * cell + pad + 8);
  canvas.height = 408;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = "10px system-ui";
  ctx.textBaseline = "middle";
  for (let a = 0; a < 20; a++) {
    ctx.fillStyle = "#527263";
    ctx.fillText(result.alphabet[a], 6, top + a * cell + 9);
    for (let i = 0; i < result.shape[0]; i++) {
      const v = Math.max(-3, Math.min(3, result.ddg[i * 20 + a])) / 3,
        t = Math.abs(v),
        base = [248, 247, 239],
        target = v < 0 ? [49, 114, 165] : [199, 111, 63];
      ctx.fillStyle = result.valid[i]
        ? `rgb(${base.map((b, j) => Math.round(b + (target[j] - b) * t)).join(",")})`
        : "#aeb6af";
      ctx.fillRect(pad + i * cell, top + a * cell, cell - 1, cell - 1);
    }
  }
  ctx.fillStyle = "#527263";
  for (
    let i = 0;
    i < result.shape[0];
    i += Math.max(1, Math.ceil(result.shape[0] / 20))
  ) {
    ctx.save();
    ctx.translate(pad + i * cell + 9, 378);
    ctx.rotate(-Math.PI / 4);
    ctx.fillText(
      `${result.residues[i].chain}:${result.residues[i].number}${result.residues[i].ins}`,
      0,
      0,
    );
    ctx.restore();
  }
  $("summary").textContent =
    `${result.shape[0]} residues · ${result.timingMs.toFixed(0)} ms`;
  for (const id of ["csv", "fasta", "source"]) $(id).disabled = false;
  renderTable();
}
$("run").onclick = async () => {
  if (busy) return;
  setBusy(true);
  $("progress").value = 0;
  $("status").textContent = "Preparing prediction…";
  try {
    sourceText = $("structure").value;
    const chains = $("chains").value.trim()
      ? $("chains")
          .value.split(",")
          .map((c) => c.trim())
      : undefined;
    const opts = { chains, mode: $("mode").value };
    if (opts.mode === "targeted") opts.mutations = mutations();
    const load = {
      baseUrl: new URL($("base").value, location.href).href,
      precision: $("precision").value,
      cache: $("cache").value,
    };
    const key = JSON.stringify(load);
    if (loadedKey !== key) {
      loadedKey = "";
      await call("init", load);
      loadedKey = key;
    }
    result = await call("predict", opts, sourceText);
    render();
    $("progress").value = 100;
    $("status").textContent =
      `Done. ${(result.memory.residentBytes / 1e6).toFixed(1)} MB GPU weights; no temporary buffers retained.`;
  } catch (e) {
    $("status").textContent = e.message;
  } finally {
    setBusy(false);
  }
};
$("cancel").onclick = () => {
  worker.postMessage({ command: "cancel" });
  $("status").textContent = "Cancelling…";
};
$("release").onclick = async () => {
  await call("destroy");
  loadedKey = "";
  $("status").textContent = "GPU buffers released.";
};
$("mode").onchange = () => {
  $("mutation-field").hidden = $("mode").value !== "targeted";
};
$("file").onchange = async () => {
  const file = $("file").files[0];
  if (file) {
    $("structure").value = await file.text();
    sourceName = file.name;
  }
};
$("example").onclick = async () => {
  $("structure").value = await (await fetch("./example.pdb")).text();
  sourceName = "example.pdb";
  $("status").textContent =
    "Example loaded: 12-residue 2OCJ fragment, chain A.";
};
$("cys").onchange = renderTable;
$("heatmap").onmousemove = (e) => {
  if (!result) return;
  const r = e.target.getBoundingClientRect(),
    i = Math.floor(
      (((e.clientX - r.left) * e.target.width) / r.width - pad) / cell,
    ),
    a = Math.floor(
      (((e.clientY - r.top) * e.target.height) / r.height - top) / cell,
    );
  if (i < 0 || i >= result.shape[0] || a < 0 || a >= 20) return;
  const residue = result.residues[i];
  $("hover").textContent =
    `${residue.chain}:${residue.number}${residue.ins} · ${residue.aa} → ${result.alphabet[a]} · ${result.ddg[i * 20 + a].toFixed(4)} kcal/mol${residue.structuralValid ? "" : " · incomplete backbone"}${result.valid[i] ? "" : " · gap/unknown, not scored for ranking"}`;
};
function download(text, name, type) {
  const url = URL.createObjectURL(new Blob([text], { type })),
    a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.hidden = true;
  document.body.append(a);
  a.click();
  setTimeout(() => {
    a.remove();
    URL.revokeObjectURL(url);
  }, 60000);
}
$("csv").onclick = () =>
  download(toCSV(result), "thermompnn-ddg.csv", "text/csv");
$("fasta").onclick = () =>
  download(toFASTA(result), "thermompnn-input.fasta", "text/plain");
$("source").onclick = () => download(sourceText, sourceName, "text/plain");
startWorker();
