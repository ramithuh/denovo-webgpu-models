import { createEvoDiffEngine } from "../webgpu/index.js";
const el = (id) => document.getElementById(id);
let engine,
  loadedPrecision,
  controller,
  sequence = "",
  active = false;
el("mode").onchange = () => {
  el("motifBox").hidden = el("mode").value !== "motif";
};
function progress(p) {
  if (p.phase === "weights") {
    el("progress").value = p.loaded / p.total;
    el("status").textContent =
      `Loading weights: ${Math.round((p.loaded / p.total) * 100)}%`;
  }
  if (p.phase === "sampling") {
    el("progress").value = p.completed / p.total;
    el("status").textContent =
      `Generating residue ${p.completed} of ${p.total}`;
  }
}
el("run").onclick = async () => {
  active = true;
  controller = new AbortController();
  el("run").disabled = true;
  el("cancel").disabled = false;
  el("release").disabled = true;
  el("download").disabled = true;
  el("progress").value = 0;
  try {
    const precision = el("precision").value;
    if (!engine || loadedPrecision !== precision) {
      engine?.destroy();
      engine = null;
      el("status").textContent = "Loading model…";
      engine = await createEvoDiffEngine({
        baseUrl: new URL("../weights/oa_dm_38m", location.href).href,
        precision,
        signal: controller.signal,
        onProgress: progress,
      });
      loadedPrecision = precision;
    }
    const options = {
      seed: Number(el("seed").value),
      signal: controller.signal,
      onProgress: progress,
    };
    if (el("mode").value === "motif")
      options.fixedSequence = el("motif").value.trim();
    else options.length = Number(el("length").value);
    const out = await engine.generate(options);
    sequence = out.sequence;
    el("sequence").textContent = sequence;
    el("status").textContent =
      `Done. ${out.tokens.length} residues in ${(out.timingMs / 1000).toFixed(2)} s.`;
    el("download").disabled = false;
    el("progress").value = 1;
  } catch (e) {
    el("status").textContent = controller.signal.aborted
      ? "Cancelled."
      : `Error: ${e.message}`;
  } finally {
    active = false;
    el("run").disabled = false;
    el("cancel").disabled = true;
    el("release").disabled = false;
    controller = null;
  }
};
el("cancel").onclick = () => controller?.abort();
el("release").onclick = () => {
  if (active) return;
  engine?.destroy();
  engine = null;
  loadedPrecision = null;
  el("status").textContent = "GPU buffers released.";
};
el("download").onclick = () => {
  const text = `>EvoDiff_OA_DM_38M seed=${el("seed").value}\n${sequence.match(/.{1,80}/g).join("\n")}\n`;
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain" })),
    a = document.createElement("a");
  a.href = url;
  a.download = "evodiff.fasta";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
addEventListener("pagehide", () => {
  controller?.abort();
  engine?.destroy();
});
