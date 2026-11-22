import { assert } from "./parity.js";
export async function demoTest() {
  const frame = document.createElement("iframe");
  frame.src = "/demo/index.html";
  document.body.appendChild(frame);
  await new Promise((resolve, reject) => {
    frame.onload = resolve;
    frame.onerror = reject;
  });
  const win = frame.contentWindow,
    doc = frame.contentDocument,
    $ = (id) => doc.getElementById(id);
  const until = async (fn) => {
    const start = performance.now();
    while (!fn()) {
      if ($("status").textContent.startsWith("Error:"))
        throw Error($("status").textContent);
      if (performance.now() - start > 120000)
        throw Error("Demo timeout: " + $("status").textContent);
      await new Promise((r) => setTimeout(r, 30));
    }
  };
  try {
    $("length").value = "16";
    $("steps").value = "3";
    $("precision").value = "int8";
    $("run").click();
    await until(() => $("status").textContent.startsWith("Done."));
    assert(!$("download").disabled, "PDB enabled");
    let blob;
    const create = win.URL.createObjectURL,
      click = win.HTMLAnchorElement.prototype.click;
    win.URL.createObjectURL = (b) => {
      blob = b;
      return create.call(win.URL, b);
    };
    win.HTMLAnchorElement.prototype.click = () => {};
    try {
      $("download").click();
      const pdb = await blob.text();
      assert(
        pdb.split("\n").filter((l) => l.startsWith("ATOM")).length === 16,
        "Demo PDB atoms",
      );
    } finally {
      win.URL.createObjectURL = create;
      win.HTMLAnchorElement.prototype.click = click;
    }
    const pixels = $("view").getContext("2d").getImageData(0, 0, 720, 720).data;
    assert(
      pixels.some((x) => x > 0),
      "Backbone preview",
    );
    $("steps").value = "100";
    $("run").click();
    await until(() => $("status").textContent.startsWith("Generating"));
    $("cancel").click();
    await until(() => $("status").textContent === "Cancelled.");
    $("release").click();
    await until(() => $("status").textContent === "GPU buffers released.");
    return {
      generation: true,
      preview: true,
      pdbDownload: true,
      cancel: true,
      release: true,
    };
  } finally {
    frame.remove();
  }
}
