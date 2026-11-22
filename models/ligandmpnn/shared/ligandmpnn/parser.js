/** Holo PDB/mmCIF parsing. First model, positive occupancy, blank/A altloc. */
export const ALPHABET = "ACDEFGHIKLMNPQRSTVWYX";
const RES = Object.fromEntries(
  "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR UNK"
    .split(" ")
    .map((x, i) => [x, ALPHABET[i]]),
);
RES.MSE = "M";
export const ATOMS =
  "N CA C CB O CG CG1 CG2 OG OG1 SG CD CD1 CD2 ND1 ND2 OD1 OD2 SD CE CE1 CE2 CE3 NE NE1 NE2 OE1 OE2 CH2 NH1 NH2 OH CZ CZ2 CZ3 NZ OXT".split(
    " ",
  );
const ELEMENTS =
  "X H HE LI BE B C N O F NE NA MG AL SI P S CL AR K CA SC TI V CR MN FE CO NI CU ZN GA GE AS SE BR KR RB SR Y ZR NB MO TC RU RH PD AG CD IN SN SB TE I XE CS BA LA CE PR ND PM SM EU GD TB DY HO ER TM YB LU HF TA W RE OS IR PT AU HG TL PB BI PO AT RN FR RA AC TH PA U NP PU AM CM BK CF ES FM MD NO LR RF DB SG BH HS MT DS RG CN NH FL MC LV TS OG".split(
    " ",
  );
function pdbAtoms(text) {
  const out = [];
  let model = null;
  for (const l of text.split(/\r?\n/)) {
    if (l.startsWith("MODEL ")) {
      if (model !== null) break;
      model = l.slice(10).trim();
    }
    if (l.startsWith("ENDMDL")) break;
    if (!/^(ATOM  |HETATM)/.test(l)) continue;
    const name = l.slice(12, 16).trim();
    // PDB atom-name alignment disambiguates protein CA from calcium.
    const inferred =
      l[12] === " "
        ? name.replace(/^\d/, "")[0]
        : name.replace(/^\d/, "").slice(0, 2);
    out.push({
      name,
      alt: l[16].trim(),
      res: l.slice(17, 20).trim(),
      chain: l[21].trim(),
      seq: l.slice(22, 26).trim(),
      ins: l[26].trim(),
      xyz: [+l.slice(30, 38), +l.slice(38, 46), +l.slice(46, 54)],
      occ: l.slice(54, 60).trim() === "" ? 1 : +l.slice(54, 60),
      element: l.slice(76, 78).trim().toUpperCase() || inferred.toUpperCase(),
    });
  }
  return out;
}
// CIF tokenizer supports quoted values, comments and semicolon multiline text.
function tokenize(text) {
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
      let end = text.indexOf("\n;", i + 1);
      if (end < 0) throw Error("Unterminated CIF text");
      out.push({ v: text.slice(i + 1, end), quoted: true });
      i = end + 2;
      continue;
    }
    if (text[i] === '"' || text[i] === "'") {
      const q = text[i++],
        start = i;
      while (
        i < text.length &&
        !(text[i] === q && (i + 1 === text.length || /\s/.test(text[i + 1])))
      )
        i++;
      if (i === text.length) throw Error("Unterminated CIF quote");
      out.push({ v: text.slice(start, i++), quoted: true });
      continue;
    }
    const start = i;
    while (i < text.length && !/\s/.test(text[i])) i++;
    out.push({ v: text.slice(start, i), quoted: false });
  }
  return out;
}
function cifAtoms(text) {
  const ts = tokenize(text),
    out = [];
  let model;
  const missing = (v) => v === undefined || v === "." || v === "?";
  for (let i = 0; i < ts.length; i++) {
    if (ts[i].quoted || ts[i].v !== "loop_") continue;
    const cols = [];
    while (ts[i + 1]?.v.startsWith("_") && !ts[i + 1].quoted)
      cols.push(ts[++i].v);
    if (!cols.length) continue;
    const isAtom = cols[0].startsWith("_atom_site.");
    while (i + 1 < ts.length) {
      const next = ts[i + 1];
      if (!next.quoted && /^(?:_|loop_|stop_|data_|save_)/.test(next.v)) break;
      if (i + cols.length >= ts.length) throw Error("Incomplete CIF loop row");
      const row = ts.slice(i + 1, i + 1 + cols.length).map((t) => t.v);
      i += cols.length;
      if (!isAtom) continue;
      const val = (...keys) => {
        for (const k of keys) {
          const v = row[cols.indexOf("_atom_site." + k)];
          if (!missing(v)) return v;
        }
        return "";
      };
      const m = val("pdbx_PDB_model_num") || "1";
      model ??= m;
      if (m !== model) continue;
      out.push({
        name: val("auth_atom_id", "label_atom_id"),
        res: val("auth_comp_id", "label_comp_id"),
        chain: val("auth_asym_id", "label_asym_id"),
        seq: val("auth_seq_id", "label_seq_id"),
        ins: val("pdbx_PDB_ins_code"),
        alt: val("label_alt_id"),
        occ: val("occupancy") === "" ? 1 : +val("occupancy"),
        xyz: ["Cartn_x", "Cartn_y", "Cartn_z"].map((k) => +val(k)),
        element: val("type_symbol").toUpperCase(),
      });
    }
  }
  return out;
}
export function parseStructure(text, { format } = {}) {
  const atoms =
    format === "mmcif" ||
    format === "cif" ||
    (!format && /(?:^|\n)\s*(?:data_|_atom_site\.)/.test(text))
      ? cifAtoms(text)
      : pdbAtoms(text);
  const residues = new Map(),
    context = [],
    chains = new Map();
  for (const a of atoms) {
    if (a.occ <= 0 || !["", "A", "1"].includes(a.alt)) continue;
    if (!a.xyz.every(Number.isFinite))
      throw Error("Non-finite atom coordinate");
    if (!chains.has(a.chain)) chains.set(a.chain, chains.size);
    if (a.res in RES) {
      const key = JSON.stringify([a.chain, a.seq, a.ins]);
      if (!residues.has(key))
        residues.set(key, {
          chain: a.chain,
          seq: a.seq,
          ins: a.ins,
          name: a.res,
          atoms: new Map(),
        });
      const r = residues.get(key);
      const name = a.res === "MSE" && a.name === "SE" ? "SD" : a.name;
      if (!r.atoms.has(name) || a.alt === "") r.atoms.set(name, a);
    } else if (!["HOH", "WAT", "DOD"].includes(a.res)) {
      const element = ELEMENTS.indexOf(a.element);
      if (element > 1) context.push({ ...a, element });
    }
  }
  const rs = [...residues.values()].filter((r) => r.atoms.has("CA"));
  if (!rs.length) throw Error("No protein alpha carbons found");
  const L = rs.length,
    X = new Float32Array(L * 12),
    xyz_37 = new Float32Array(L * 111),
    xyz_37_m = new Float32Array(L * 37),
    mask = new Float32Array(L),
    S = new Int32Array(L),
    R_idx = new Int32Array(L),
    chain_labels = new Int32Array(L);
  rs.forEach((r, i) => {
    R_idx[i] = Number(r.seq);
    if (!Number.isInteger(Number(r.seq)))
      throw Error("Non-integer author residue number: " + r.seq);
    chain_labels[i] = chains.get(r.chain);
    S[i] = ALPHABET.indexOf(RES[r.name]);
    ATOMS.forEach((a, j) => {
      const atom = r.atoms.get(a);
      if (atom && j < 36) {
        xyz_37.set(atom.xyz, i * 111 + j * 3);
        xyz_37_m[i * 37 + j] = 1;
      }
    });
    ["N", "CA", "C", "O"].forEach((a, j) => {
      if (r.atoms.has(a)) X.set(r.atoms.get(a).xyz, i * 12 + j * 3);
    });
    mask[i] = ["N", "CA", "C", "O"].every((a) => r.atoms.has(a)) ? 1 : 0;
  });
  const Y = new Float32Array(Math.max(1, context.length) * 3),
    Y_t = new Int32Array(Math.max(1, context.length)),
    Y_m = new Float32Array(Math.max(1, context.length));
  context.forEach((a, i) => {
    Y.set(a.xyz, i * 3);
    Y_t[i] = a.element;
    Y_m[i] = 1;
  });
  return {
    L,
    X,
    xyz_37,
    xyz_37_m,
    mask,
    S,
    R_idx,
    chain_labels,
    Y,
    Y_t,
    Y_m,
    residues: rs.map((r) => ({
      chain: r.chain,
      number: Number(r.seq),
      insertionCode: r.ins,
      token: r.chain + r.seq + r.ins,
    })),
    context,
  };
}
