import { diffStats } from "../shared/weights.js";
export async function fixture(name, read, base = "") {
  const m = JSON.parse(
      new TextDecoder().decode(await read(base + "fixtures/" + name + ".json")),
    ),
    data = await read(base + m.file);
  const hash = Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", data)),
    (x) => x.toString(16).padStart(2, "0"),
  ).join("");
  if (hash !== m.sha256) throw Error("Fixture checksum mismatch");
  const input = { length: m.length },
    output = {};
  for (const [name, r] of Object.entries(m.tensors)) {
    const v = new Float32Array(data.slice(r.offset, r.offset + r.count * 4));
    if (name.startsWith("input."))
      input[name.slice(6)] = ["so3_t", "r3_t"].includes(name.slice(6))
        ? v[0]
        : v;
    else output[name.slice(7)] = v;
  }
  return { meta: m, input, output };
}
export async function verify(engine, f) {
  const start = performance.now();
  const actual =
    f.meta.kind === "forward"
      ? await engine.forward(f.input)
      : await engine.sample({
          initial: f.input,
          steps: f.meta.steps,
          keepTrajectory: true,
        });
  const metrics = {};
  for (const [k, ref] of Object.entries(f.output)) {
    let v = actual[k];
    if (k.startsWith("step")) {
      const [s, n] = k.split(".");
      v = actual.trajectory[Number(s.slice(4))][n];
    }
    metrics[k] = diffStats(v, ref);
    if (metrics[k].relRms >= 1e-4)
      throw Error(`${f.meta.name} ${k} relRMS ${metrics[k].relRms} >=1e-4`);
  }
  return {
    name: f.meta.name,
    length: f.meta.length,
    steps: f.meta.steps ?? 1,
    ms: performance.now() - start,
    metrics,
  };
}
