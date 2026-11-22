import { constants } from "./constants.js";
import { makeGeometry, dihedral, sub } from "./geometry.js";
import { createRng } from "./model.js";
const clean = (x) => (x === "." || x === "?" ? "" : x);
const number = (x, what) => {
  const n = Number(x);
  if (!Number.isFinite(n) || x == null || x === "")
    throw Error(`Invalid ${what}`);
  return n;
};
function* cifTokens(text) {
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
      if (end < 0) throw Error("Unterminated CIF multiline value");
      yield text.slice(start, end);
      i = end + 2;
      continue;
    }
    if (text[i] === "'" || text[i] === '"') {
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
      if (i === text.length) throw Error("Unterminated CIF quote");
      yield text.slice(start, i++);
      continue;
    }
    const start = i;
    while (i < text.length && !/\s/.test(text[i])) i++;
    yield text.slice(start, i);
  }
}
function cifAtoms(text) {
  const ts = Array.from(cifTokens(text)),
    atoms = [];
  for (let i = 0; i < ts.length; i++)
    if (ts[i] === "loop_") {
      const names = [];
      while (ts[i + 1]?.startsWith("_")) names.push(ts[++i]);
      if (!names.length) throw Error("Empty CIF loop");
      const rows = [];
      while (
        i + 1 < ts.length &&
        !/^(?:_|loop_|data_|save_|stop_)/.test(ts[i + 1])
      )
        rows.push(ts[++i]);
      if (!names[0].startsWith("_atom_site.")) continue;
      if (rows.length % names.length) throw Error("Incomplete atom_site row");
      for (let r = 0; r < rows.length; r += names.length) {
        const row = Object.fromEntries(
            names.map((n, j) => [n.slice(11), rows[r + j]]),
          ),
          get = (a, b) => clean(row[a] ?? "") || clean(row[b] ?? "");
        atoms.push({
          record: row.group_PDB ?? "ATOM",
          serial: row.id,
          name: get("auth_atom_id", "label_atom_id"),
          resname: get("auth_comp_id", "label_comp_id"),
          chain: get("auth_asym_id", "label_asym_id"),
          resnum: get("auth_seq_id", "label_seq_id"),
          icode: get("pdbx_PDB_ins_code"),
          segment: "",
          altloc: get("label_alt_id"),
          model: get("pdbx_PDB_model_num") || "1",
          element: row.type_symbol,
          xyz: ["Cartn_x", "Cartn_y", "Cartn_z"].map((k) => number(row[k], k)),
          beta: Number(row.B_iso_or_equiv) || 0,
        });
      }
    }
  return atoms;
}
function pdbAtoms(text) {
  const atoms = [];
  let model = "1";
  for (const line of text.split(/\r?\n/)) {
    const record = line.slice(0, 6).trim();
    if (record === "MODEL") {
      model = line.slice(10, 14).trim();
      continue;
    }
    if (!["ATOM", "HETATM"].includes(record)) continue;
    const name = line.slice(12, 16).trim(),
      element =
        line.slice(76, 78).trim() ||
        name.replace(/[0-9]/g, "").slice(0, line[12] === " " ? 1 : 2);
    atoms.push({
      record,
      serial: line.slice(6, 11).trim(),
      name,
      resname: line.slice(17, 20).trim(),
      chain: line.slice(21, 22).trim(),
      resnum: line.slice(22, 26).trim(),
      icode: line.slice(26, 27).trim(),
      segment: line.slice(72, 76).trim(),
      altloc: line.slice(16, 17).trim(),
      model,
      element,
      xyz: [30, 38, 46].map((i) =>
        number(line.slice(i, i + 8).trim(), "PDB coordinate"),
      ),
      beta: Number(line.slice(60, 66)) || 0,
    });
  }
  return atoms;
}
export function parseStructure(text, { format, useWater = false } = {}) {
  if (typeof text !== "string" || !text.trim())
    throw Error("Structure text is empty");
  const atoms =
    ["cif", "mmcif"].includes(format) || (!format && /^\s*data_\S*/m.test(text))
      ? cifAtoms(text)
      : pdbAtoms(text);
  if (!atoms.length) throw Error("No atoms in structure");
  const model = atoms[0].model,
    chains = new Map();
  for (const a of atoms) {
    if (a.model !== model || !["", "A"].includes(a.altloc)) continue;
    const ck = a.segment + "|" + a.chain;
    let residues = chains.get(ck);
    if (!residues) {
      residues = new Map();
      chains.set(ck, residues);
    }
    const key = a.resnum + "|" + a.icode;
    let res = residues.get(key);
    if (!res) {
      res = { ...a, atoms: new Map(), id: `${a.chain}:${a.resnum}${a.icode}` };
      residues.set(key, res);
    }
    if (res.resname !== a.resname)
      throw Error(`Conflicting residue names at ${res.id}`);
    if (!res.atoms.has(a.name)) res.atoms.set(a.name, a);
  }
  const residues = [],
    ligands = [];
  let group = 0;
  for (const chain of chains.values())
    for (const res of chain.values()) {
      if (res.resname === "HOH" && !useWater) continue;
      if (["N", "CA", "C", "O"].every((n) => res.atoms.has(n)))
        residues.push(res);
      else {
        for (const a of res.atoms.values()) ligands.push({ ...a, group });
        group++;
      }
    }
  if (!residues.length) throw Error("No complete protein backbone residues");
  return { residues, ligands };
}
export function featurize(structure, weights, config, opts = {}) {
  const geom = makeGeometry(weights),
    residues = structure.residues,
    n = residues.length,
    backbone = new Float32Array(n * 15),
    phiPsi = new Float32Array(n * 2).fill(NaN),
    sequence = new Int32Array(n),
    chi = new Float32Array(n * 4).fill(NaN),
    fixed = new Int32Array(n);
  const xyz = (r, a) => r?.atoms.get(a)?.xyz ?? [NaN, NaN, NaN],
    distance = (a, b) => Math.hypot(...sub(a, b));
  for (let i = 0; i < n; i++) {
    const r = residues[i],
      p = residues[i - 1],
      next = residues[i + 1];
    sequence[i] = constants.residueNames.indexOf(r.resname);
    if (sequence[i] < 0) sequence[i] = 20;
    const same = (a) => a && a.chain === r.chain && a.segment === r.segment;
    if (same(p) && distance(xyz(p, "CA"), xyz(r, "CA")) < 4.1)
      phiPsi[2 * i] = dihedral(
        xyz(p, "C"),
        xyz(r, "N"),
        xyz(r, "CA"),
        xyz(r, "C"),
      );
    if (same(next) && distance(xyz(next, "CA"), xyz(r, "CA")) < 4.1)
      phiPsi[2 * i + 1] = dihedral(
        xyz(r, "N"),
        xyz(r, "CA"),
        xyz(r, "C"),
        xyz(next, "N"),
      );
    backbone.set(
      geom.idealize(
        ["N", "CA", "C"].map((a) => xyz(r, a)),
        phiPsi[2 * i + 1],
      ),
      i * 15,
    );
    const names = constants.heavyAtomNames[sequence[i]],
      points = names.map((a) => xyz(r, a).map(Math.fround));
    [0, 1, 3, 4, 2].forEach(
      (k, j) =>
        (points[j] = Array.from(
          backbone.slice(i * 15 + k * 3, i * 15 + k * 3 + 3),
        )),
    );
    const ids = geom.lookup("aa_to_chi_angle_atom_index", sequence[i]);
    for (let j = 0; j < 4; j++)
      chi[4 * i + j] = dihedral(
        ...ids.slice(j * 4, j * 4 + 4).map((k) => points[k] ?? [NaN, NaN, NaN]),
      );
    fixed[i] = Number(
      (opts.fixedPositions ?? []).includes(r.id) ||
        (opts.designChains && !opts.designChains.includes(r.chain)) ||
        (opts.fixBeta &&
          Math.abs(
            Math.max(...Array.from(r.atoms.values(), (a) => a.beta)) - 1,
          ) < 1e-5),
    );
  }
  const lig = structure.ligands,
    ligandCoords = Float32Array.from(lig.flatMap((a) => a.xyz)),
    ligandNodes = new Float32Array(lig.length * 26),
    atomicNumbers = lig.map((a) => {
      const el = a.element[0]?.toUpperCase() + a.element.slice(1).toLowerCase();
      const z =
        el === "D"
          ? 1
          : Number(
              Object.keys(constants.elements).find(
                (k) => constants.elements[k] === el,
              ),
            );
      if (!z) throw Error(`Unknown ligand element ${a.element}`);
      return z;
    });
  atomicNumbers.forEach((z, i) => {
    ligandNodes[
      i * 26 +
        weights["ligand_featurizer.atomic_number_idx_to_period_idx"].data[z - 1]
    ] = 1;
    ligandNodes[
      i * 26 +
        7 +
        weights["ligand_featurizer.atomic_number_idx_to_group_idx"].data[z - 1]
    ] = 1;
  });
  const ca = Array.from({ length: n }, (_, i) =>
      Array.from(backbone.slice(i * 15 + 3, i * 15 + 6)),
    ),
    lc = Array.from({ length: lig.length }, (_, i) =>
      Array.from(ligandCoords.slice(i * 3, i * 3 + 3)),
    ),
    g = config.model.graph_structure;
  const nearest = (points, target, filter = () => true) =>
    points
      .map((p, i) => ({ i, d: distance(p, target) }))
      .filter((x) => filter(x.i))
      .sort((a, b) => a.d - b.d || a.i - b.i);
  const pp = [[], []],
    ll = [[], []],
    lp = [[], []];
  ca.forEach((p, s) =>
    nearest(ca, p)
      .slice(0, g.pr_pr_knn_graph_k)
      .forEach((x) => {
        pp[0].push(x.i);
        pp[1].push(s);
      }),
  );
  lc.forEach((p, s) =>
    nearest(lc, p, (i) => lig[i].group === lig[s].group)
      .slice(0, g.lig_lig_knn_graph_k)
      .forEach((x) => {
        ll[0].push(x.i);
        ll[1].push(s);
      }),
  );
  ca.forEach((p, s) => {
    const sorted = nearest(lc, p);
    if (sorted[0]?.d < g.lig_pr_distance_cutoff)
      sorted.slice(0, g.lig_pr_knn_graph_k).forEach((x) => {
        lp[0].push(x.i);
        lp[1].push(s);
      });
  });
  const point = (r, a) =>
    Array.from(backbone.slice(r * 15 + a * 3, r * 15 + a * 3 + 3));
  const ppDistances = Float32Array.from(
    pp[0].flatMap((src, e) =>
      Array.from({ length: 25 }, (_, j) =>
        distance(point(src, Math.floor(j / 5)), point(pp[1][e], j % 5)),
      ),
    ),
  );
  const lpDistances = Float32Array.from(
    lp[0].flatMap((src, e) =>
      Array.from({ length: 5 }, (_, j) =>
        distance(lc[src], point(lp[1][e], j)),
      ),
    ),
  );
  const llDistances = Float32Array.from(
    ll[0].map((src, e) => distance(lc[src], lc[ll[1][e]])),
  );
  if (opts.firstShellOnly) {
    const built = geom.build(backbone, sequence, chi, { hydrogens: false }),
      stride = built.atomsPerResidue;
    for (let i = 0; i < n; i++) {
      const gly = sequence[i] === 7 || sequence[i] === 20,
        points = gly
          ? [ca[i]]
          : Array.from({ length: stride - 4 }, (_, j) =>
              Array.from(
                built.coords.slice(
                  (i * stride + j + 4) * 3,
                  (i * stride + j + 5) * 3,
                ),
              ),
            );
      const contact = lc.some(
        (a, l) =>
          atomicNumbers[l] !== 1 &&
          points.some((p) => distance(p, a) < (gly ? 5.3 : 5)),
      );
      if (!contact) fixed[i] = 1;
    }
  }
  const rng = createRng(opts.seed ?? 42),
    keys = Array.from({ length: n }, (_, i) => ({
      i,
      key: rng() + (fixed[i] ? 0 : 1),
    }));
  keys.sort((a, b) => a.key - b.key);
  return {
    length: n,
    backbone,
    phiPsi,
    sequence,
    chi,
    fixed,
    ligandCoords,
    ligandNodes,
    pp,
    lp,
    ll,
    ppDistances,
    lpDistances,
    llDistances,
    decodingOrder: keys.map((x) => x.i),
    rng,
    structure,
  };
}
export function writePdb(batch, result, built) {
  let serial = 1;
  const mapping = new Map(),
    used = new Set(),
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  const key = (r) => JSON.stringify([r.segment, r.chain]);
  const records = [...batch.structure.residues, ...batch.structure.ligands];
  for (const r of records)
    if (!mapping.has(key(r)) && r.chain.length <= 1 && !used.has(r.chain)) {
      mapping.set(key(r), r.chain);
      used.add(r.chain);
    }
  for (const r of records)
    if (!mapping.has(key(r))) {
      const id = Array.from(alphabet).find((c) => !used.has(c));
      if (id === undefined) throw Error("Too many chains for PDB output");
      mapping.set(key(r), id);
      used.add(id);
    }
  const chainId = (r) => mapping.get(key(r));
  const lines = Array.from(
      mapping,
      ([original, id]) =>
        `REMARK 950 CHAIN ${id || "<blank>"} ORIGINAL_SEGMENT_CHAIN ${original}`,
    ),
    line = (record, name, resname, chain, resnum, icode, xyz, element) => {
      if (chain.length > 1)
        throw Error("PDB output requires single-character chain IDs");
      if (icode.length > 1 || String(resnum).length > 4)
        throw Error("Residue identifier cannot be represented in PDB");
      if (xyz.some((x) => !Number.isFinite(x))) return;
      const a =
        name.length < 4 && element.length === 1
          ? " " + name.padEnd(3)
          : name.padEnd(4);
      lines.push(
        `${record.padEnd(6)}${String(serial++).padStart(5)} ${a} ${resname.padStart(3)} ${chain || " "}${String(resnum).padStart(4)}${icode || " "}   ${xyz.map((v) => v.toFixed(3).padStart(8)).join("")}  1.00  0.00          ${element.toUpperCase().padStart(2)}`,
      );
    };
  batch.structure.residues.forEach((r, i) => {
    const seq = result.sequence[i],
      names = constants.atomNames[seq];
    names.forEach((a, j) =>
      line(
        "ATOM",
        a,
        constants.residueNames[seq],
        chainId(r),
        r.resnum,
        r.icode,
        Array.from(
          built.coords.slice(
            (i * built.atomsPerResidue + j) * 3,
            (i * built.atomsPerResidue + j + 1) * 3,
          ),
        ),
        a[0],
      ),
    );
    if (built.nhCoords)
      line(
        "ATOM",
        "H",
        constants.residueNames[seq],
        chainId(r),
        r.resnum,
        r.icode,
        Array.from(built.nhCoords.slice(i * 3, i * 3 + 3)),
        "H",
      );
  });
  for (const a of batch.structure.ligands)
    line(
      "HETATM",
      a.name,
      a.resname,
      chainId(a),
      a.resnum,
      a.icode,
      a.xyz,
      a.element,
    );
  lines.push("END");
  return lines.join("\n") + "\n";
}
