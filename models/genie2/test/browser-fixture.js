export async function loadFixture(precision, index = 0) {
  const response = await fetch(`/fixtures/oracle_${precision}.json`);
  if (!response.ok) throw Error("Required oracle missing");
  const j = await response.json(),
    c = j.cases[index],
    res = await fetch("/" + c.file);
  if (!res.ok) throw Error("Required binary oracle missing");
  const b = await res.arrayBuffer();
  const hash = Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", b)),
    (v) => v.toString(16).padStart(2, "0"),
  ).join("");
  if (hash !== c.sha256) throw Error("Fixture checksum mismatch");
  const get = (k) => {
    const t = c.tensors[k];
    if (!t) throw Error(`Missing tensor ${k}`);
    return new Float32Array(b, t.offset, t.byteLength / 4).slice();
  };
  const features = Object.fromEntries(
    Object.keys(c.tensors)
      .filter((k) => k.startsWith("features."))
      .map((k) => [k.slice(9), get(k)]),
  );
  return { c, get, features };
}
export async function log(message) {
  await fetch("/log", { method: "POST", body: JSON.stringify(message) });
}
