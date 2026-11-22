export const ALPHABET = "ACDEFGHIKLMNPQRSTVWYX";
const aa3 = Object.fromEntries(
  "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR"
    .split(" ")
    .map((a, i) => [a, ALPHABET[i]]),
);
const atoms = ["N", "CA", "C", "O"];
const chainOrder = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
  .split("")
  .concat(Array.from({ length: 300 }, (_, i) => String(i)));

// CIF 1.1 tokenization: quoted values, comments, and semicolon multiline fields.
export function cifTokens(text) {
  const out = [];
  let i = 0;
  while (i < text.length) {
    if (/\s/.test(text[i])) {
      i++;
      continue;
    }
    if (text[i] === "#") {
      while (i < text.length && text[i] !== "\n") i++;
      continue;
    }
    if (text[i] === ";" && (i === 0 || text[i - 1] === "\n")) {
      const start = ++i,
        end = text.indexOf("\n;", i);
      if (end < 0) throw new Error("Unterminated CIF text field");
      out.push(text.slice(start, end));
      i = end + 2;
      continue;
    }
    if (text[i] === '"' || text[i] === "'") {
      const quote = text[i++],
        start = i;
      while (
        i < text.length &&
        !(
          text[i] === quote &&
          (i + 1 === text.length || /\s/.test(text[i + 1]))
        )
      )
        i++;
      if (i === text.length) throw new Error("Unterminated CIF quote");
      out.push(text.slice(start, i++));
      continue;
    }
    const start = i;
    while (i < text.length && !/\s/.test(text[i])) i++;
    out.push(text.slice(start, i));
  }
  return out;
}
function cifRecords(text) {
  const t = cifTokens(text),
    rows = [];
  for (let i = 0; i < t.length; ) {
    if (t[i++] !== "loop_") continue;
    const cols = [];
    while (t[i]?.startsWith("_")) cols.push(t[i++]);
    if (!cols.length) throw new Error("Empty CIF loop");
    const atom = cols[0].startsWith("_atom_site.");
    const col = Object.fromEntries(
      cols.map((v, j) => [v.replace("_atom_site.", ""), j]),
    );
    while (i < t.length && !/^(loop_|stop_|data_|save_|_)/.test(t[i])) {
      if (i + cols.length > t.length)
        throw new Error("Incomplete CIF loop row");
      const values = t.slice(i, i + cols.length);
      i += cols.length;
      if (!atom) continue;
      const get = (...names) => {
        for (const name of names) {
          const v = values[col[name]];
          if (v !== undefined && v !== "." && v !== "?") return v;
        }
        return "";
      };
      rows.push({
        record: get("group_PDB"),
        atom: get("auth_atom_id", "label_atom_id"),
        name: get("auth_comp_id", "label_comp_id"),
        chain: get("auth_asym_id", "label_asym_id"),
        number: Number(get("auth_seq_id", "label_seq_id")),
        ins: get("pdbx_PDB_ins_code"),
        xyz: ["Cartn_x", "Cartn_y", "Cartn_z"].map((k) =>
          get(k) === "" ? NaN : Number(get(k)),
        ),
      });
    }
  }
  return rows;
}
function pdbRecords(text) {
  const rows = [];
  for (const l of text.split(/\r?\n/)) {
    if (!l.startsWith("ATOM") && !l.startsWith("HETATM")) continue;
    rows.push({
      record: l.slice(0, 6).trim(),
      atom: l.slice(12, 16).trim(),
      name: l.slice(17, 20),
      chain: l.slice(21, 22),
      number: Number(l.slice(22, 26).trim()),
      ins: l.slice(26, 27).trim(),
      xyz: [30, 38, 46].map((i) =>
        l.slice(i, i + 8).trim() === "" ? NaN : Number(l.slice(i, i + 8)),
      ),
    });
  }
  return rows;
}
export function parseStructure(
  text,
  { format, chains, maxLength = 2048 } = {},
) {
  if (!Number.isSafeInteger(maxLength) || maxLength < 1 || maxLength > 2048)
    throw new Error("maxLength must be an integer between 1 and 2048");
  if (typeof text !== "string" || !text.trim())
    throw new Error("Structure text is empty");
  format = (
    format ??
    (/^\s*(data_|#.*\n.*data_)/.test(text) || text.includes("_atom_site.")
      ? "cif"
      : "pdb")
  ).toLowerCase();
  if (!["pdb", "cif", "mmcif"].includes(format))
    throw new Error("Expected PDB or mmCIF");
  if (
    chains !== undefined &&
    (!Array.isArray(chains) ||
      !chains.length ||
      new Set(chains).size !== chains.length ||
      chains.some((x) => typeof x !== "string"))
  )
    throw new Error("chains must be a nonempty array of distinct IDs");
  const byChain = new Map();
  for (const r of format === "pdb" ? pdbRecords(text) : cifRecords(text)) {
    if (r.record === "HETATM" && r.name === "MSE") {
      r.record = "ATOM";
      r.name = "MET";
    }
    if (r.record !== "ATOM") continue;
    if (chains && !chains.includes(r.chain)) continue;
    if (
      !Number.isSafeInteger(r.number) ||
      r.xyz.some((x) => !Number.isFinite(x))
    )
      throw new Error("Invalid residue number or coordinates");
    if (!byChain.has(r.chain)) byChain.set(r.chain, new Map());
    const residues = byChain.get(r.chain),
      key = `${r.number}:${r.ins}`;
    if (!residues.has(key)) residues.set(key, { ...r, atoms: new Map() });
    const res = residues.get(key);
    if (!res.atoms.has(r.atom)) res.atoms.set(r.atom, r.xyz);
  }
  const ids =
    chains ??
    [...byChain.keys()].sort((a, b) => {
      const ai = chainOrder.indexOf(a),
        bi = chainOrder.indexOf(b);
      return ai >= 0 && bi >= 0
        ? ai - bi
        : ai >= 0
          ? -1
          : bi >= 0
            ? 1
            : a.localeCompare(b, "en");
    });
  if (!ids.length) throw new Error("No protein ATOM records");
  const residues = [],
    coords = [],
    mask = [],
    atomMask = [],
    S = [],
    chain = [],
    residue_idx = [];
  for (const [ci, id] of ids.entries()) {
    const map = byChain.get(id);
    if (!map) throw new Error(`Chain not found: ${id}`);
    const items = [...map.values()].sort(
      (a, b) =>
        a.number - b.number || (a.ins < b.ins ? -1 : a.ins > b.ins ? 1 : 0),
    );
    const lo = items[0].number,
      hi = items.at(-1).number;
    if (hi - lo + 1 + residues.length > maxLength)
      throw new Error(
        `Structure exceeds maxLength ${maxLength} (including gaps)`,
      );
    let j = 0;
    for (let number = lo; number <= hi; number++) {
      const here = [];
      while (j < items.length && items[j].number === number)
        here.push(items[j++]);
      if (!here.length)
        here.push({ number, ins: "", name: "GAP", atoms: new Map() });
      for (const r of here) {
        const index = residues.length,
          aa = aa3[r.name] ?? "X";
        let valid = true;
        for (const a of atoms) {
          const xyz = r.atoms.get(a);
          if (!xyz) valid = false;
          atomMask.push(Number(!!xyz));
          coords.push(...(xyz ?? [0, 0, 0]));
        }
        residues.push({
          index,
          chain: id,
          number,
          ins: r.ins,
          aa,
          missing: r.name === "GAP",
          structuralValid: valid,
        });
        mask.push(Number(valid));
        S.push(ALPHABET.indexOf(aa));
        chain.push(ci + 1);
        residue_idx.push(100 * ci + index);
        if (residues.length > maxLength)
          throw new Error(`Structure exceeds maxLength ${maxLength}`);
      }
    }
  }
  return {
    length: residues.length,
    sequence: residues.map((r) => r.aa).join(""),
    residues,
    chains: ids,
    X: new Float32Array(coords),
    atomMask: new Uint8Array(atomMask),
    S: new Int32Array(S),
    mask: new Float32Array(mask),
    chain: new Int32Array(chain),
    residue_idx: new Int32Array(residue_idx),
  };
}
export function resolveMutations(structure, mutations) {
  if (!Array.isArray(mutations) || !mutations.length)
    throw new Error("targeted mode requires a nonempty mutations array");
  return mutations.map((m) => {
    if (!m || typeof m !== "object")
      throw new Error("Mutation must be an object");
    let index = m.index;
    if (index === undefined) {
      if (typeof m.chain !== "string" || !Number.isSafeInteger(m.number))
        throw new Error("Provide index or chain/number/ins");
      index = structure.residues.findIndex(
        (r) =>
          r.chain === m.chain &&
          r.number === m.number &&
          r.ins === (m.ins ?? ""),
      );
    }
    if (!Number.isInteger(index) || index < 0 || index >= structure.length)
      throw new Error("Mutation residue not found");
    const r = structure.residues[index];
    if (r.missing || r.aa === "X")
      throw new Error("Cannot target a gap or unknown residue");
    if (m.wildtype !== undefined && m.wildtype !== r.aa)
      throw new Error(`Wild-type mismatch at ${r.chain}:${r.number}${r.ins}`);
    if (
      typeof m.mutation !== "string" ||
      m.mutation.length !== 1 ||
      !ALPHABET.slice(0, 20).includes(m.mutation)
    )
      throw new Error("Expected one canonical mutant amino acid");
    return {
      ...r,
      wildtype: r.aa,
      mutation: m.mutation,
      aminoAcidIndex: ALPHABET.indexOf(m.mutation),
    };
  });
}
