import { ALPHABET, resolveMutations } from "./structure.js";
export function prepareInput(text, options, parse) {
  if (options.mode !== undefined && !["ssm", "targeted"].includes(options.mode))
    throw new Error("mode must be ssm or targeted");
  const structure = parse(text, options);
  const mutations =
    options.mode === "targeted" || options.mutations !== undefined
      ? resolveMutations(structure, options.mutations)
      : null;
  return { structure, mutations };
}
export function resultFrom(structure, ddg, mutations, timingMs, intermediates) {
  for (const x of ddg)
    if (!Number.isFinite(x)) throw new Error("Nonfinite model output");
  const scores = mutations
    ? Float32Array.from(mutations, (m) => ddg[m.index * 20 + m.aminoAcidIndex])
    : undefined;
  return {
    ddg,
    shape: [structure.length, 20],
    alphabet: ALPHABET.slice(0, 20),
    sequence: structure.sequence,
    residues: structure.residues,
    mask: structure.mask,
    valid: Uint8Array.from(
      structure.residues,
      (r) => !r.missing && r.aa !== "X",
    ),
    ...(mutations ? { mutations, scores } : {}),
    timingMs,
    ...(intermediates ? { intermediates } : {}),
  };
}
export function bestMutants(result, { includeCys = true } = {}) {
  return result.residues
    .filter((r) => result.valid[r.index])
    .map((r) => {
      let best = -1;
      for (let a = 0; a < 20; a++) {
        if (!includeCys && ALPHABET[a] === "C") continue;
        if (
          best < 0 ||
          result.ddg[r.index * 20 + a] < result.ddg[r.index * 20 + best]
        )
          best = a;
      }
      return {
        ...r,
        mutation: ALPHABET[best],
        ddG: result.ddg[r.index * 20 + best],
      };
    });
}
export function centrality(structure, { radius = 10, basisAtom = "CA" } = {}) {
  if (!(radius > 0) || !Number.isFinite(radius))
    throw new Error("radius must be positive and finite");
  const atom = ["N", "CA", "C", "O"].indexOf(basisAtom);
  if (atom < 0) throw new Error("Unknown backbone basis atom");
  // Upstream compute_centrality runs per chain, counts self, then subtracts one.
  // An absent basis atom therefore returns -1 (not zero).
  return Int32Array.from(structure.residues, (_, i) => {
    if (!structure.atomMask[i * 4 + atom]) return -1;
    let n = 0;
    for (let j = 0; j < structure.length; j++) {
      if (
        i === j ||
        structure.chain[i] !== structure.chain[j] ||
        !structure.atomMask[j * 4 + atom]
      )
        continue;
      let d = 0;
      for (let c = 0; c < 3; c++)
        d +=
          (structure.X[i * 12 + atom * 3 + c] -
            structure.X[j * 12 + atom * 3 + c]) **
          2;
      if (d < radius ** 2) n++;
    }
    return n;
  });
}
const csv = (x) => `"${String(x).replaceAll('"', '""')}"`;
export function toCSV(result) {
  const rows = ["chain,residue,insertion_code,index,wildtype,mutation,ddG"];
  const mutations =
    result.mutations ??
    result.residues
      .filter((r) => result.valid[r.index])
      .flatMap((r) =>
        [...result.alphabet].map((mutation, aminoAcidIndex) => ({
          ...r,
          mutation,
          aminoAcidIndex,
        })),
      );
  for (const m of mutations)
    rows.push(
      [
        m.chain,
        m.number,
        m.ins,
        m.index,
        m.aa,
        m.mutation,
        result.ddg[m.index * 20 + m.aminoAcidIndex],
      ]
        .map(csv)
        .join(","),
    );
  return rows.join("\n") + "\n";
}
export function toFASTA(result) {
  return [...new Set(result.residues.map((r) => r.chain))]
    .map(
      (c) =>
        `>ThermoMPNN_chain_${c}\n${result.residues
          .filter((r) => r.chain === c)
          .map((r) => r.aa)
          .join("")}\n`,
    )
    .join("");
}
