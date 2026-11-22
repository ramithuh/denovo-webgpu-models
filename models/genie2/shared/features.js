export const RESTYPES = "ARNDCQEGHILKMFPSTWYV";
export function buildFeatures({
  length,
  chainLengths,
  motifs = [],
  ...overrides
} = {}) {
  if (chainLengths) length = chainLengths.reduce((a, b) => a + b, 0);
  if (!Number.isSafeInteger(length) || length < 3)
    throw new RangeError("length must be an integer >= 3");
  const L = length,
    f = { length: L };
  for (const [k, n] of Object.entries({
    aatype: L * 20,
    atom_positions: L * 3,
    residue_mask: L,
    residue_index: L,
    chain_index: L,
    fixed_sequence_mask: L,
    fixed_structure_mask: L * L,
    fixed_group: L,
    interface_mask: L,
  }))
    f[k] = new Float32Array(n);
  f.residue_mask.fill(1);
  let offset = 0;
  for (const [chain, n] of (chainLengths ?? [L]).entries()) {
    if (!Number.isInteger(n) || n < 3)
      throw new RangeError("Each chain must have >= 3 residues");
    for (let i = 0; i < n; i++) {
      f.residue_index[offset + i] = i;
      f.chain_index[offset + i] = chain;
    }
    offset += n;
  }
  for (const [index, m] of motifs.entries()) {
    const positions = m.coordinates ?? m.frames?.trans;
    if (!positions || positions.length % 3)
      throw Error("Motif coordinates must be flat C-alpha xyz values");
    const indices =
        m.indices ??
        Array.from(
          { length: positions.length / 3 },
          (_, i) => (m.start ?? 0) + i,
        ),
      group = m.group ?? index + 1;
    if (
      indices.length * 3 !== positions.length ||
      !Number.isInteger(group) ||
      group <= 0
    )
      throw Error("Invalid motif indices or group");
    for (const [j, i] of indices.entries()) {
      if (!Number.isInteger(i) || i < 0 || i >= L || f.fixed_sequence_mask[i])
        throw Error("Overlapping or out-of-range motif");
      f.fixed_sequence_mask[i] = 1;
      f.fixed_group[i] = group;
      f.atom_positions.set(positions.slice(j * 3, j * 3 + 3), i * 3);
      const aa = RESTYPES.indexOf(m.sequence?.[j] ?? "A");
      if (aa < 0) throw Error("Unknown motif amino acid");
      f.aatype[i * 20 + aa] = 1;
    }
  }
  for (let i = 0; i < L; i++)
    for (let j = 0; j < L; j++)
      f.fixed_structure_mask[i * L + j] = Number(
        f.fixed_sequence_mask[i] &&
          f.fixed_sequence_mask[j] &&
          f.fixed_group[i] === f.fixed_group[j],
      );
  for (const k of Object.keys(f))
    if (k in overrides) {
      if (k === "length") continue;
      const a = overrides[k];
      if (a.length !== f[k].length) throw Error(`Wrong shape for ${k}`);
      f[k] = Float32Array.from(a);
    }
  if (!("fixed_structure_mask" in overrides))
    for (let i = 0; i < L; i++)
      for (let j = 0; j < L; j++)
        f.fixed_structure_mask[i * L + j] = Number(
          f.fixed_sequence_mask[i] &&
            f.fixed_sequence_mask[j] &&
            f.fixed_group[i] === f.fixed_group[j],
        );
  for (const [k, a] of Object.entries(f)) {
    if (k === "length") continue;
    if (a.some((x) => !Number.isFinite(x)))
      throw Error(`Nonfinite feature ${k}`);
  }
  for (const k of [
    "residue_mask",
    "fixed_sequence_mask",
    "fixed_structure_mask",
  ])
    if (f[k].some((x) => x !== 0 && x !== 1))
      throw Error(`Invalid binary mask ${k}`);
  const n = f.residue_mask.reduce((a, b) => a + b, 0);
  if (n < 3 || f.residue_mask.some((v, i) => v !== Number(i < n)))
    throw Error(
      "Residue mask must be a contiguous valid prefix of at least 3 residues",
    );
  return f;
}
// Uniform over valid segment-length assignments, with bounded dynamic programming.
export function sampleScaffold(
  { segments, minLength = 3, maxLength = 256 },
  rng = Math.random,
) {
  if (!Array.isArray(segments) || !segments.length)
    throw Error("Missing scaffold segments");
  const choices = segments.map((s) =>
    s.motif
      ? [(s.motif.coordinates ?? s.motif.frames?.trans)?.length / 3]
      : Array.from({ length: s.max - s.min + 1 }, (_, i) => s.min + i),
  );
  if (
    choices.some(
      (a) => !a.length || a.some((n) => !Number.isSafeInteger(n) || n < 0),
    )
  )
    throw Error("Invalid scaffold range");
  if (
    !Number.isInteger(maxLength) ||
    maxLength < minLength ||
    maxLength > 100000
  )
    throw Error("Invalid total range");
  const memo = new Map();
  const count = (i, n) => {
    if (n > maxLength) return 0;
    if (i === segments.length) return Number(n >= minLength);
    const key = i + ":" + n;
    if (memo.has(key)) return memo.get(key);
    const sum = choices[i].reduce((s, k) => s + count(i + 1, n + k), 0);
    memo.set(key, sum);
    return sum;
  };
  const total = count(0, 0);
  if (!Number.isFinite(total))
    throw Error("Scaffold range combinations exceed numeric capacity");
  if (!total) throw Error("Scaffold ranges have no valid total length");
  let n = 0;
  const motifs = [],
    lengths = [];
  for (let i = 0; i < segments.length; i++) {
    let r = rng() * count(i, n),
      chosen = choices[i].at(-1);
    for (const k of choices[i]) {
      r -= count(i + 1, n + k);
      if (r < 0) {
        chosen = k;
        break;
      }
    }
    lengths.push(chosen);
    if (segments[i].motif) motifs.push({ ...segments[i].motif, start: n });
    n += chosen;
  }
  return { length: n, motifs, segmentLengths: lengths };
}
