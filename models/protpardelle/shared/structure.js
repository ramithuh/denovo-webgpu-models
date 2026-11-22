import { constants as C } from "./constants.js";
export function tokenize(sequence) {
  const normalized = sequence.replace(/\s+/g, "").toUpperCase();
  if (!normalized.length) throw new Error("Empty sequence");
  return Int32Array.from(normalized, (aa) => {
    const i = C.restypes.indexOf(aa);
    if (i < 0) throw new Error(`Unsupported amino acid ${aa}`);
    return i;
  });
}
function pdbRows(text) {
  const rows = [];
  let model;
  for (const line of text.split(/\r?\n/)) {
    const record = line.slice(0, 6).trim();
    if (record === "MODEL") {
      if (model !== undefined) break;
      model = line.slice(10).trim();
    }
    if (record === "ENDMDL") break;
    if (record !== "ATOM" && record !== "HETATM") continue;
    rows.push({
      record,
      atom: line.slice(12, 16).trim(),
      alt: line.slice(16, 17).trim(),
      resname: line.slice(17, 20).trim(),
      chain: line.slice(21, 22),
      resid: Number(line.slice(22, 26)),
      icode: line.slice(26, 27).trim(),
      xyz: [30, 38, 46].map((i) => Number(line.slice(i, i + 8))),
      occupancy: Number(line.slice(54, 60)),
      bfactor: Number(line.slice(60, 66)),
    });
  }
  return rows;
}
function cifTokens(text) {
  const tokens = [];
  let i = 0,
    lineStart = true;
  while (i < text.length) {
    if (/\s/.test(text[i])) {
      lineStart = text[i] === "\n" || (lineStart && text[i] !== "\r");
      i++;
      continue;
    }
    if (text[i] === "#") {
      while (i < text.length && text[i] !== "\n") i++;
      continue;
    }
    if (text[i] === ";" && lineStart) {
      const end = text.indexOf("\n;", i + 1);
      if (end < 0) throw new Error("Unterminated mmCIF text");
      tokens.push(text.slice(i + 1, end));
      i = end + 2;
      lineStart = false;
      continue;
    }
    lineStart = false;
    if (text[i] === "'" || text[i] === '"') {
      const quote = text[i++],
        start = i;
      while (
        i < text.length &&
        !(
          text[i] === quote &&
          (/\s/.test(text[i + 1] ?? " ") || i + 1 === text.length)
        )
      )
        i++;
      if (i === text.length) throw new Error("Unterminated mmCIF quote");
      tokens.push(text.slice(start, i++));
    } else {
      const start = i;
      while (i < text.length && !/\s/.test(text[i])) i++;
      tokens.push(text.slice(start, i));
    }
  }
  return tokens;
}
function cifRows(text) {
  const tokens = cifTokens(text),
    rows = [];
  let model;
  for (let i = 0; i < tokens.length; i++)
    if (tokens[i] === "loop_") {
      const fields = [];
      while (tokens[i + 1]?.startsWith("_")) fields.push(tokens[++i]);
      if (!fields.length) throw new Error("Empty mmCIF loop");
      const atom = fields[0].startsWith("_atom_site.");
      while (
        i + 1 < tokens.length &&
        !tokens[i + 1].startsWith("_") &&
        !["loop_", "stop_"].includes(tokens[i + 1]) &&
        !tokens[i + 1].startsWith("data_")
      ) {
        const record = {};
        for (const field of fields) {
          if (i + 1 >= tokens.length) throw new Error("Truncated mmCIF row");
          record[field] = tokens[++i];
        }
        if (!atom) continue;
        const get = (k) => record["_atom_site." + k],
          current = get("pdbx_PDB_model_num") ?? "1";
        model ??= current;
        if (model !== current) continue;
        // Upstream explicitly uses label_seq_id, not auth_seq_id.
        if (get("label_seq_id") === "." || get("label_seq_id") === "?")
          continue;
        rows.push({
          record: get("group_PDB"),
          atom: get("label_atom_id"),
          alt: [".", "?"].includes(get("label_alt_id"))
            ? ""
            : get("label_alt_id"),
          resname: get("label_comp_id"),
          chain: get("auth_asym_id") ?? get("label_asym_id"),
          resid: Number(get("label_seq_id")),
          icode: [".", "?"].includes(get("pdbx_PDB_ins_code"))
            ? ""
            : get("pdbx_PDB_ins_code"),
          xyz: ["Cartn_x", "Cartn_y", "Cartn_z"].map((k) => Number(get(k))),
          occupancy: Number(get("occupancy")),
          bfactor: Number(get("B_iso_or_equiv")),
        });
      }
    }
  return rows;
}
export function parseStructure(
  text,
  { format = "pdb", chainId, chainResidueGap = 200 } = {},
) {
  if (!["pdb", "cif", "mmcif"].includes(format))
    throw new Error("Unknown structure format");
  const rows = format === "pdb" ? pdbRows(text) : cifRows(text),
    chains = new Map();
  for (const row of rows) {
    if (chainId !== undefined && row.chain !== chainId) continue;
    if (!row.xyz.every(Number.isFinite) || !Number.isInteger(row.resid))
      throw new Error("Invalid atom coordinates or residue number");
    if (!chains.has(row.chain)) chains.set(row.chain, new Map());
    const residues = chains.get(row.chain),
      id = `${row.record === "HETATM" ? "H_" + row.resname : " "}|${row.resid}|${row.icode}`;
    if (!residues.has(id)) residues.set(id, { ...row, atoms: new Map() });
    const residue = residues.get(id),
      old = residue.atoms.get(row.atom);
    if (!old || row.occupancy > old.occupancy) residue.atoms.set(row.atom, row);
  }
  const residues = [],
    hetero = [];
  for (const chain of chains.values())
    for (const residue of chain.values()) {
      const aa =
        C.restype3to1[residue.resname] ??
        (residue.resname === "UNK" ? "G" : null);
      if (!aa) {
        hetero.push({ ...residue, atoms: [...residue.atoms.values()] });
        continue;
      }
      const atoms = [...residue.atoms.values()].filter((atom) =>
        C.atomTypes.includes(atom.atom),
      );
      if (atoms.length) residues.push({ ...residue, aa, atoms });
    }
  if (!residues.length)
    throw new Error("Structure has no recognized protein atoms");
  const chainIds = [...new Set(residues.map((r) => r.chain))].sort(),
    n = residues.length;
  if (chainIds.length > 62)
    throw new Error("Upstream supports at most 62 chains");
  const coords = new Float32Array(n * 111),
    atomMask = new Float32Array(n * 37),
    bFactors = new Float32Array(n * 37),
    aatype = new Int32Array(n),
    origAatype = new Int32Array(n).fill(20),
    residueIndex = new Float32Array(n),
    originalIndex = new Int32Array(n),
    chainIndex = new Int32Array(n);
  for (let r = 0; r < n; r++) {
    const residue = residues[r];
    aatype[r] = C.restypes.indexOf(residue.aa);
    originalIndex[r] = residue.resid;
    residueIndex[r] = residue.resid;
    chainIndex[r] = chainIds.indexOf(residue.chain);
    for (const atom of residue.atoms) {
      const index = C.atomTypes.indexOf(atom.atom);
      coords.set(atom.xyz, r * 111 + index * 3);
      atomMask[r * 37 + index] = 1;
      bFactors[r * 37 + index] = atom.bfactor;
    }
  }
  const min = Math.min(...residueIndex);
  for (let i = 0; i < n; i++) residueIndex[i] -= min - 1;
  for (let chain = 1; chain < chainIds.length; chain++) {
    const indices = Array.from({ length: n }, (_, i) => i).filter(
        (i) => chainIndex[i] === chain,
      ),
      first = indices[0];
    const delta =
      chainResidueGap === 0
        ? 1 - residueIndex[first]
        : residueIndex[(first - 1 + n) % n] +
          chainResidueGap -
          residueIndex[first];
    for (const i of indices) residueIndex[i] += delta;
  }
  return {
    coords,
    atomMask,
    bFactors,
    aatype,
    origAatype,
    residueIndex,
    originalIndex,
    chainIndex,
    chainIds,
    seqMask: new Float32Array(n).fill(1),
    residues: residues.map(({ atoms, ...r }) => r),
    hetero,
  };
}
export function toPDB({ coords, aatype, chainIndex, residueIndex, atomMask }) {
  const chains =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
    lines = [];
  let serial = 1;
  for (let r = 0; r < aatype.length; r++)
    for (let a = 0; a < 37; a++)
      if (atomMask?.[r * 37 + a] ?? C.atom37Masks[aatype[r]][a]) {
        const name = C.atomTypes[a],
          res = C.restype1to3[C.restypes[aatype[r]]],
          chain = chains[chainIndex?.[r] ?? 0];
        if (!chain) throw new Error("PDB chain limit exceeded");
        const xyz = Array.from(
          coords.slice(r * 111 + a * 3, r * 111 + a * 3 + 3),
          (v) => v.toFixed(3).padStart(8),
        ).join("");
        lines.push(
          `ATOM  ${String(serial++).padStart(5)} ${name.padStart(3).padEnd(4)} ${res} ${chain}${String(residueIndex?.[r] ?? r + 1).padStart(4)}    ${xyz}  1.00  0.00          ${name[0].padStart(2)}  `,
        );
      }
  return lines.join("\n") + "\nEND\n";
}
