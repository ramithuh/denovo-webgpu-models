import { createEsmcEngine } from "../webgpu/index.js";
import { ALPHABET, VOCAB } from "../shared/index.js";
const $ = (id) => document.getElementById(id),
  seqs = {
    trp: "NLYIQWLKDGGPSSGRPPPS",
    ubi: "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
  };
let engine,
  engineKey,
  controller,
  result,
  busy = false;
function controls() {
  $("run").disabled = busy;
  $("cancel").disabled = !busy;
  $("precision").disabled = busy;
  $("cache").disabled = busy;
}
function progress(p) {
  if (p.total) {
    $("progress").value = p.loaded / p.total;
    $("status").textContent =
      p.phase === "mask-sweep"
        ? `Mask ${p.loaded} / ${p.total}`
        : `Loading weights: ${((100 * p.loaded) / p.total).toFixed(0)}%`;
  } else if (p.phase === "cache-warning")
    $("status").textContent =
      "Browser storage unavailable; continuing with the download.";
}
function release() {
  controller?.abort();
  engine?.destroy();
  engine = undefined;
  engineKey = undefined;
  $("status").textContent = "GPU released.";
}
$("release").onclick = release;
$("cancel").onclick = () => controller?.abort();
window.addEventListener("pagehide", release);
$("preset").onchange = () => {
  $("sequence").value = seqs[$("preset").value];
  $("positions").value = `2,${$("sequence").value.length - 1}`;
};
$("run").onclick = async () => {
  busy = true;
  controls();
  controller = new AbortController();
  $("progress").value = 0;
  try {
    const key = $("precision").value + ":" + $("cache").value;
    if (engineKey !== key) {
      engine?.destroy();
      engine = undefined;
      $("status").textContent = "Loading ESMC-300M…";
      engine = await createEsmcEngine({
        baseUrl: "../weights/esmc-300m",
        precision: $("precision").value,
        cache: $("cache").value,
        signal: controller.signal,
        onProgress: progress,
      });
      engineKey = key;
    }
    $("status").textContent = "Computing…";
    const sequence = $("sequence").value.replace(/\s/g, "").toUpperCase(),
      mode = $("mode").value,
      hiddenLayer =
        $("hiddenLayer").value === ""
          ? undefined
          : Number($("hiddenLayer").value);
    result = await engine.run(sequence, {
      mode,
      hiddenLayer,
      positions: $("positions")
        .value.split(",")
        .map((s) => Number(s.trim()) - 1),
      signal: controller.signal,
      onProgress: progress,
    });
    $("progress").value = 1;
    $("status").textContent =
      `Complete · ${(result.timingMs / 1000).toFixed(2)} s`;
    $("csv").disabled = false;
    $("fasta").disabled = false;
    render();
  } catch (e) {
    $("status").textContent =
      e.name === "AbortError" ? "Cancelled." : e.message;
  } finally {
    busy = false;
    controls();
  }
};
function render() {
  const r = result,
    L = r.residues.length,
    canvas = $("plot"),
    ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  $("hover").textContent = "";
  $("summary").textContent = `${L} residues · ${r.mode}`;
  if (r.mode === "embeddings") {
    const v = r.pooled,
      w = canvas.width / v.length;
    ctx.fillStyle = "#d5ed91";
    const mx = Math.max(...v.map(Math.abs), 1e-8);
    v.forEach((x, i) =>
      ctx.fillRect(i * w, 190, Math.max(w, 1), (-x / mx) * 170),
    );
    $("result").textContent =
      `Per-residue embeddings: ${L} × 960\nPooled embedding: 960 values\n${r.hiddenState ? "Selected hidden state included in CSV." : ""}`;
  } else if (r.mode === "naturalness") {
    const v = r.wtLogProbs,
      mx = Math.max(...v.map((x) => -x), 1e-8);
    ctx.fillStyle = "#d5ed91";
    v.forEach((x, i) =>
      ctx.fillRect(
        (i * 700) / L + 1,
        360 + (x / mx) * 340,
        Math.max(700 / L - 2, 1),
        (-x / mx) * 340,
      ),
    );
    $("result").textContent =
      `Naturalness (mean masked log p): ${r.naturalness.toFixed(6)}\nPseudo-perplexity: ${r.pseudoPerplexity.toFixed(6)}`;
  } else {
    const v =
        r.mode === "logits"
          ? Float32Array.from(
              { length: L * 20 },
              (_, i) =>
                r.logits[
                  (Math.floor(i / 20) + 1) * 64 +
                    VOCAB.indexOf(ALPHABET[i % 20])
                ],
            )
          : r.llr,
      scale = Math.max(...v.map(Math.abs), 1);
    for (let p = 0; p < L; p++)
      for (let a = 0; a < 20; a++) {
        const x = v[p * 20 + a] / scale;
        ctx.fillStyle =
          x >= 0
            ? `rgb(${40 + 170 * x},${70 + 165 * x},${60 + 75 * x})`
            : `rgb(${40 - 170 * x},${70 - 15 * x},${60 - 15 * x})`;
        ctx.fillRect((p * 700) / L, (a * 380) / 20, Math.ceil(700 / L), 19);
      }
    $("result").textContent = r.filledSequence
      ? `Filled sequence:\n${r.filledSequence}`
      : "Columns: residues. Rows: " +
        ALPHABET +
        (r.mode === "logits"
          ? "\nCanonical MLM logits shown; all 64 logits are in the CSV."
          : "\nHover to inspect a mutation score.");
  }
}
$("plot").onmousemove = (e) => {
  if (
    !result?.llr ||
    result.mode === "embeddings" ||
    result.mode === "naturalness"
  )
    return;
  const rect = e.target.getBoundingClientRect(),
    p = Math.min(
      result.residues.length - 1,
      Math.floor(
        ((e.clientX - rect.left) / rect.width) * result.residues.length,
      ),
    ),
    a = Math.min(19, Math.floor(((e.clientY - rect.top) / rect.height) * 20));
  const value =
    result.mode === "logits"
      ? result.logits[(p + 1) * 64 + VOCAB.indexOf(ALPHABET[a])]
      : result.llr[p * 20 + a];
  $("hover").textContent =
    `${result.residues[p]}${p + 1}${ALPHABET[a]} · ${result.mode === "logits" ? "logit" : "LLR"} ${value.toFixed(5)}`;
};
function download(name, text, type) {
  const u = URL.createObjectURL(new Blob([text], { type })),
    a = document.createElement("a");
  a.href = u;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(u), 1000);
}
$("fasta").onclick = () =>
  download(
    "esmc-sequence.fasta",
    `>ESMC_${result.mode}\n${result.filledSequence ?? result.residues}\n`,
    "text/plain",
  );
$("csv").onclick = () => {
  const r = result,
    rows = [];
  if (r.mode === "embeddings") {
    rows.push(
      "position,residue," +
        Array.from({ length: 960 }, (_, i) => "embedding_" + i).join(","),
    );
    for (let p = 0; p < r.residues.length; p++)
      rows.push(
        [
          p + 1,
          r.residues[p],
          ...r.embeddings.subarray(p * 960, (p + 1) * 960),
        ].join(","),
      );
    rows.push(["pooled", "", ...r.pooled].join(","));
    if (r.hiddenState)
      for (let p = 0; p < r.residues.length; p++)
        rows.push(
          [
            "hidden_" + (p + 1),
            r.residues[p],
            ...r.hiddenState.subarray((p + 1) * 960, (p + 2) * 960),
          ].join(","),
        );
  } else if (r.mode === "logits") {
    rows.push(
      "position,residue," +
        Array.from({ length: 64 }, (_, i) => VOCAB[i] ?? "reserved_" + i).join(
          ",",
        ),
    );
    for (let p = 0; p < r.residues.length; p++)
      rows.push(
        [
          p + 1,
          r.residues[p],
          ...r.logits.subarray((p + 1) * 64, (p + 2) * 64),
        ].join(","),
      );
  } else if (r.mode === "naturalness") {
    rows.push("position,residue,masked_wt_log_probability");
    for (let p = 0; p < r.residues.length; p++)
      rows.push([p + 1, r.residues[p], r.wtLogProbs[p]].join(","));
  } else {
    rows.push("position,wild_type," + [...ALPHABET].join(","));
    for (let p = 0; p < r.residues.length; p++)
      rows.push(
        [p + 1, r.residues[p], ...r.llr.subarray(p * 20, (p + 1) * 20)].join(
          ",",
        ),
      );
  }
  download(`esmc-${r.mode}.csv`, rows.join("\n") + "\n", "text/csv");
};
