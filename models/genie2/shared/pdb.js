import { RESTYPES } from "./features.js";
const names = [
  "ALA",
  "ARG",
  "ASN",
  "ASP",
  "CYS",
  "GLN",
  "GLU",
  "GLY",
  "HIS",
  "ILE",
  "LEU",
  "LYS",
  "MET",
  "PHE",
  "PRO",
  "SER",
  "THR",
  "TRP",
  "TYR",
  "VAL",
];
export function parsePDB(text) {
  const residues = [],
    seen = new Set();
  let model = false;
  for (const line of text.split(/\r?\n/)) {
    if (line.startsWith("MODEL")) {
      if (model) break;
      model = true;
    }
    if (line.startsWith("ENDMDL")) break;
    if (
      !/^ATOM  /.test(line) ||
      line.slice(12, 16).trim() !== "CA" ||
      ![" ", "A", ""].includes(line[16] ?? "")
    )
      continue;
    const chain = line[21] ?? " ",
      number = Number(line.slice(22, 26)),
      insertion = line[26] ?? " ",
      key = `${chain}:${number}:${insertion}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const coordinates = [30, 38, 46].map((i) => Number(line.slice(i, i + 8)));
    if (
      !Number.isFinite(number) ||
      coordinates.some((x) => !Number.isFinite(x))
    )
      throw Error("Invalid PDB coordinate");
    const aa = names.indexOf(line.slice(17, 20));
    if (aa < 0) throw Error("Unknown PDB residue " + line.slice(17, 20));
    residues.push({
      chain,
      number,
      insertion,
      aminoAcid: RESTYPES[aa],
      coordinates,
    });
  }
  if (!residues.length) throw Error("No C-alpha ATOM records");
  return {
    residues,
    sequence: residues.map((x) => x.aminoAcid).join(""),
    coordinates: Float32Array.from(residues.flatMap((x) => x.coordinates)),
  };
}
export function writePDB({
  coordinates,
  sequence,
  features,
  center = true,
} = {}) {
  if (!coordinates?.length || coordinates.length % 3)
    throw Error("Invalid C-alpha coordinates");
  const L = coordinates.length / 3,
    mean = [0, 0, 0];
  if (center)
    for (let i = 0; i < L; i++)
      for (let k = 0; k < 3; k++) mean[k] += coordinates[i * 3 + k] / L;
  const out = [
    "REMARK   Genie 2 C-alpha backbone; no side chains or sequence design.",
  ];
  for (let i = 0; i < L; i++) {
    let code = 0;
    if (features?.aatype)
      for (let k = 1; k < 20; k++)
        if (features.aatype[i * 20 + k] > features.aatype[i * 20 + code])
          code = k;
    const aa = sequence?.[i] ?? RESTYPES[code],
      name = names[RESTYPES.indexOf(aa)];
    if (!name) throw Error("Unknown residue");
    const xyz = [0, 1, 2].map((k) => coordinates[i * 3 + k] - mean[k]);
    if (xyz.some((x) => !Number.isFinite(x) || x <= -1000 || x >= 10000))
      throw Error("Coordinates exceed PDB field width");
    const chain =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"[
        features?.chain_index?.[i] ?? 0
      ];
    if (!chain) throw Error("PDB chain limit exceeded");
    const number = (features?.residue_index?.[i] ?? i) + 1;
    if (number > 9999) throw Error("PDB residue-number limit exceeded");
    out.push(
      `ATOM  ${String(i + 1).padStart(5)}  CA  ${name} ${chain}${String(number).padStart(4)}    ${xyz.map((x) => x.toFixed(3).padStart(8)).join("")}  1.00  0.00           C`,
    );
  }
  out.push("TER", "END");
  return out.join("\n") + "\n";
}
export function parseMotifSpecification(text) {
  const pdb = parsePDB(text),
    segments = [],
    groups = new Map();
  let minLength = 3,
    maxLength = 256;
  for (const line of text.split(/\r?\n/)) {
    if (line.startsWith("REMARK 999 MINIMUM TOTAL LENGTH"))
      minLength = Number(line.slice(31).trim());
    if (line.startsWith("REMARK 999 MAXIMUM TOTAL LENGTH"))
      maxLength = Number(line.slice(31).trim());
    if (!line.startsWith("REMARK 999 INPUT")) continue;
    const parts = line.slice(16).trim().split(/\s+/);
    if (/^\d+$/.test(parts[0])) {
      segments.push({ min: Number(parts[0]), max: Number(parts[1]) });
      continue;
    }
    const [chain, start, end, group = "A"] = parts,
      selected = pdb.residues.filter(
        (r) =>
          r.chain === chain &&
          r.number >= Number(start) &&
          r.number <= Number(end),
      );
    if (!selected.length) throw Error("Motif segment has no C-alpha atoms");
    if (!groups.has(group)) groups.set(group, groups.size + 1);
    segments.push({
      motif: {
        sequence: selected.map((r) => r.aminoAcid).join(""),
        coordinates: Float32Array.from(selected.flatMap((r) => r.coordinates)),
        group: groups.get(group),
      },
    });
  }
  if (!segments.length) throw Error("Missing REMARK 999 INPUT segments");
  return { segments, minLength, maxLength };
}
