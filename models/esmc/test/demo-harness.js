import { check } from "./parity.js";
const waitFor = async (predicate, ms = 180000) => {
  const start = performance.now();
  while (!predicate()) {
    if (performance.now() - start > ms) throw new Error("Demo timed out");
    await new Promise((r) => setTimeout(r, 50));
  }
};
export async function run() {
  const frame = document.createElement("iframe");
  frame.src = "/demo/index.html";
  document.body.append(frame);
  await new Promise((resolve, reject) => {
    frame.onload = resolve;
    frame.onerror = reject;
  });
  const doc = frame.contentDocument,
    $ = (id) => doc.getElementById(id),
    win = frame.contentWindow;
  const downloads = [],
    blobs = new Map(),
    create = win.URL.createObjectURL.bind(win.URL);
  win.URL.createObjectURL = (b) => {
    const u = create(b);
    blobs.set(u, b);
    return u;
  };
  win.HTMLAnchorElement.prototype.click = function () {
    downloads.push({ name: this.download, blob: blobs.get(this.href) });
  };
  const modes = [
    "embeddings",
    "logits",
    "wildtype",
    "masked",
    "naturalness",
    "fill",
  ];
  for (const mode of modes) {
    $("mode").value = mode;
    $("run").click();
    await waitFor(() => $("cancel").disabled);
    check(
      $("status").textContent.startsWith("Complete"),
      $("status").textContent,
    );
    $("csv").click();
    $("fasta").click();
    check($("summary").textContent.includes(mode), "Demo mode result missing");
  }
  check(downloads.length === 12, "Downloads missing");
  for (const d of downloads)
    check((await d.blob.text()).length > 30, `Empty download ${d.name}`);
  check(
    $("result").textContent.includes("Filled sequence"),
    "Fill result missing",
  );
  $("mode").value = "masked";
  $("run").click();
  $("cancel").click();
  await waitFor(() => $("cancel").disabled);
  check($("status").textContent === "Cancelled.", "Cancellation state missing");
  $("release").click();
  check($("status").textContent === "GPU released.", "Release failed");
  const shared = await import("/.cache/vendor-smoke/shared/index.js"),
    gpu = await import("/.cache/vendor-smoke/webgpu/index.js");
  check(
    shared.tokenize("AC").length === 4 &&
      typeof gpu.createEsmcEngine === "function",
    "Vendored module imports failed",
  );
  frame.remove();
  return {
    modes,
    downloadCount: downloads.length,
    cancelled: true,
    released: true,
    vendorImports: true,
  };
}
