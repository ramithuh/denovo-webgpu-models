const f = Math.fround;
export function schedule(steps = 16, tInit = 0.37) {
  if (!Number.isInteger(steps) || steps < 1)
    throw new RangeError("steps must be a positive integer");
  if (!(tInit >= 0 && tInit < 1))
    throw new RangeError("tInit must be in [0,1)");
  if (steps === 1) return Float32Array.of(tInit);
  const evenRound = (x) => {
    const b = Math.floor(x),
      r = x - b;
    return r === 0.5 ? b + (b % 2) : Math.round(x);
  };
  return Float32Array.from({ length: steps }, (_, i) => {
    const s = evenRound(f(30 + (i * 370) / (steps - 1)));
    return f(1 - f(10 ** f(-f(s / 400) * 2)));
  });
}
export function gaussianNoise(length, steps, seed = 5) {
  let state = seed >>> 0;
  const uniform = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let z = state;
    z = Math.imul(z ^ (z >>> 15), z | 1);
    z ^= z + Math.imul(z ^ (z >>> 7), z | 61);
    return ((z ^ (z >>> 14)) >>> 0) / 4294967296;
  };
  const a = new Float32Array(length * 3 * steps);
  for (let i = 0; i < a.length; i += 2) {
    const r = Math.sqrt(-2 * Math.log(Math.max(uniform(), 2 ** -32))),
      t = 2 * Math.PI * uniform();
    a[i] = r * Math.cos(t);
    if (i + 1 < a.length) a[i + 1] = r * Math.sin(t);
  }
  return a;
}
export function cathIndices(options = {}) {
  let ids = options.cathIndices;
  if (options.cathCode !== undefined) {
    if (!options.cathMapping)
      throw new Error(
        "Named CATH codes require the upstream cathMapping; the checkpoint does not contain it",
      );
    const parts = options.cathCode.split(".");
    ids = ["C", "A", "T"].map(
      (level, i) =>
        options.cathMapping[level]?.[parts.slice(0, i + 1).join(".")] ??
        [5, 43, 1472][i],
    );
  }
  ids = ids ?? [5, 43, 1472];
  if (
    ids.length !== 3 ||
    ids.some((x, i) => !Number.isInteger(x) || x < 0 || x >= [6, 44, 1473][i])
  )
    throw new RangeError("Invalid CATH indices");
  return ids;
}
export function validateInput(x, t) {
  if (
    !(x instanceof Float32Array) ||
    !x.length ||
    x.length % 3 ||
    x.some((v) => !Number.isFinite(v))
  )
    throw new TypeError("coordinates must be finite Float32Array [L,3]");
  if (!Number.isFinite(t) || t < 0 || t >= 1)
    throw new RangeError("time must be in [0,1)");
}
export function toPDB(coordinates, { chain = "A", units = "nm" } = {}) {
  validateInput(coordinates, 0);
  if (!/^[A-Za-z0-9]$/.test(chain))
    throw new RangeError("PDB chain must be one alphanumeric character");
  if (!["nm", "angstrom"].includes(units))
    throw new RangeError("Unknown coordinate units");
  const scale = units === "nm" ? 10 : 1,
    lines = [];
  for (let i = 0; i < coordinates.length / 3; i++) {
    if (i >= 9999) throw new RangeError("PDB residue index exceeds 9999");
    const xyz = Array.from(coordinates.subarray(i * 3, i * 3 + 3), (v) => {
      const s = (v * scale).toFixed(3);
      if (s.length > 8) throw new RangeError("PDB coordinate overflow");
      return s.padStart(8);
    }).join("");
    lines.push(
      `ATOM  ${String(i + 1).padStart(5)}  CA  GLY ${chain}${String(i + 1).padStart(4)}    ${xyz}  1.00  0.00           C  `,
    );
  }
  return lines.join("\n") + "\nTER\nEND\n";
}
