import { residues } from "./residues.js";
import { randomSource } from "./random.js";
export { randomInputs } from "./random.js";
const aa1 = "ARNDCQEGHILKMFPSTWYV",
  aa3 = [
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
const bb = ["N", "CA", "C", "O"];
export function parsePDB(pdb) {
  if (typeof pdb !== "string") throw Error("targetPDB must be PDB text");
  const out = [],
    byKey = new Map();

  for (const line of pdb.split(/\r?\n/)) {
    if (line.startsWith("ENDMDL")) break;
    if (!line.startsWith("ATOM  ")) continue;
    const alt = line[16];
    if (alt !== " " && alt !== "A") continue;
    const name = line.slice(17, 20).trim(),
      atom = line.slice(12, 16).trim();
    if (!residues[name]) throw Error(`Unsupported target residue ${name}`);
    const chain = line[21],
      id = line.slice(22, 27).trim(),
      key = chain + ":" + id;
    let r = byKey.get(key);
    if (!r) {
      r = { name, chain, id, atoms: {} };
      byKey.set(key, r);
      out.push(r);
    }
    if (!residues[name].atoms[atom]) continue;
    const xyz = [30, 38, 46].map((i) => Number(line.slice(i, i + 8).trim()));
    if (!xyz.every(Number.isFinite))
      throw Error(`Invalid PDB coordinate ${key} ${atom}`);
    r.atoms[atom] ??= xyz;
  }
  if (!out.length) throw Error("No protein ATOM records in target PDB");
  for (const r of out)
    for (const a of bb)
      if (!r.atoms[a])
        throw Error(`Missing target backbone atom ${r.chain}:${r.id} ${a}`);
  return out;
}
function targetResidues({ targetPDB, targetSequence, targetCoords, target }) {
  if (targetPDB) return parsePDB(targetPDB);
  targetSequence ??= target?.sequence;
  targetCoords ??= target?.coords;
  if (
    typeof targetSequence !== "string" ||
    !targetSequence.length ||
    !targetCoords
  )
    throw Error(
      "Binder/motif requires targetPDB or targetSequence and targetCoords",
    );
  const seq = targetSequence.toUpperCase().replace(/\s/g, ""),
    coords = Array.from(targetCoords.flat?.(Infinity) ?? targetCoords);
  if (coords.length !== seq.length * 12 || !coords.every(Number.isFinite))
    throw Error("Expected N, CA, C, O coordinates for every target residue");
  return Array.from(seq, (a, i) => {
    const name = aa3[aa1.indexOf(a)];
    if (!name) throw Error(`Unsupported amino acid ${a}`);
    return {
      name,
      chain: "B",
      id: String(i + 1),
      atoms: Object.fromEntries(
        bb.map((n, j) => [n, coords.slice(i * 12 + j * 3, i * 12 + j * 3 + 3)]),
      ),
    };
  });
}
export function createFeatures(options = {}) {
  const {
    mode = "unconditional",
    L = options.length ?? 16,
    seed = 42,
  } = options;
  if (!["unconditional", "binder", "motif"].includes(mode))
    throw Error(`Unsupported design mode: ${mode}`);
  if (!Number.isSafeInteger(L) || L < 1)
    throw Error("L must be a positive integer");
  const targets = mode === "unconditional" ? [] : targetResidues(options);
  let slots;
  if (mode === "motif") {
    const indices = options.motifIndices;
    if (
      !Array.isArray(indices) ||
      indices.length !== targets.length ||
      new Set(indices).size !== indices.length ||
      indices.some((i) => !Number.isSafeInteger(i) || i < 0 || i >= L)
    )
      throw Error(
        "motifIndices must map every target residue to a unique zero-based scaffold position within L",
      );
    slots = Array(L).fill(null);
    indices.forEach((i, j) => (slots[i] = targets[j]));
  } else slots = [...Array(L).fill(null), ...targets];
  const N = slots.length,
    chains = new Map(),
    targetAtoms = slots.map((r) =>
      r ? bb.concat(Object.keys(r.atoms).filter((a) => !bb.includes(a))) : null,
    );
  const count = targetAtoms.reduce((sum, a) => sum + (a?.length ?? 14), 0),
    A = Math.ceil(count / 32) * 32;
  // Guard dimensions before typed array allocation. Hardware limits are checked at upload.
  if (!Number.isSafeInteger(A * N) || A * N > 0x3fffffff)
    throw Error("Input exceeds addressable tensor size");
  const f = new Map(),
    make = (name, shape, value = 0) => {
      const data = new Float32Array(shape.reduce((a, b) => a * b, 1));
      if (value) data.fill(value);
      f.set("f:" + name, { shape, data });
      return data;
    };
  const tok = {};
  for (const name of [
    "token_index",
    "feature_residue_index",
    "feature_asym_id",
    "entity_id",
    "sym_id",
    "design_mask",
    "modified",
    "cyclic",
    "mol_type",
    "binding_type",
    "ss_type",
    "deletion_mean",
  ])
    tok[name] = make(name, [1, N]);
  make("method_feature", [1, N], 1);
  make("token_pad_mask", [1, N], 1);
  const rt = make("res_type", [1, N, 33]),
    profile = make("profile", [1, N, 33]);
  const msa = make("msa", [1, 1, N]),
    paired = make("msa_paired", [1, 1, N]);
  make("msa_mask", [1, 1, N], 1);
  make("has_deletion", [1, 1, N]);
  make("deletion_value", [1, 1, N]);
  make("token_bonds", [1, N, N, 1]);
  make("type_bonds", [1, N, N]);
  make("contact_threshold", [1, N, N]);
  const contact = make("contact_conditioning", [1, N, N, 5]);
  for (let i = 0; i < N * N; i++) contact[i * 5] = 1;
  const td = make("token_distance_mask", [1, N, N]),
    cc = make("center_coords", [1, N, 3]);
  const pos = make("ref_pos", [1, A, 3]),
    coords = make("coords", [1, 1, A, 3]),
    charge = make("ref_charge", [1, A]),
    element = make("ref_element", [1, A, 128]),
    chars = make("ref_atom_name_chars", [1, A, 4, 64]),
    uid = make("ref_space_uid", [1, A]);
  const atomMask = make("atom_pad_mask", [1, A]),
    resolved = make("atom_resolved_mask", [1, A]),
    mapping = make("atom_to_token", [1, A, N]),
    bbmap = make("token_to_bb4_atoms", [1, N, 4, A]);
  const tr = make("template_restype", [1, 1, N, 33]);
  for (let i = 0; i < N; i++) tr[i * 33] = 1;
  for (const name of [
    "template_mask",
    "template_mask_cb",
    "template_mask_frame",
    "visibility_ids",
  ])
    make(name, [1, 1, N]);
  for (const name of ["template_ca", "template_cb", "template_frame_t"])
    make(name, [1, 1, N, 3]);
  make("template_frame_rot", [1, 1, N, 3, 3]);
  const rng = randomSource(seed),
    atoms = [];
  let index = 0;
  const chainCounters = new Map();
  for (let t = 0; t < N; t++) {
    const r = slots[t],
      designed = !r,
      chem = designed ? null : residues[r.name],
      names = designed
        ? [...bb, ...Array.from({ length: 10 }, (_, i) => "FL" + (i + 4))]
        : targetAtoms[t];
    let chain = 0,
      resIndex = t;
    if (!designed && mode !== "motif") {
      if (!chains.has(r.chain)) chains.set(r.chain, chains.size + 1);
      chain = chains.get(r.chain);
      resIndex = chainCounters.get(chain) ?? 0;
      chainCounters.set(chain, resIndex + 1);
    }
    tok.token_index[t] = t;
    tok.feature_residue_index[t] = resIndex;
    tok.feature_asym_id[t] = chain;
    tok.entity_id[t] = chain;
    tok.design_mask[t] = designed ? 1 : 0;
    const id = designed ? 22 : chem.id;
    rt[t * 33 + id] = profile[t * 33 + id] = 1;
    msa[t] = id;
    paired[t] = designed ? 0 : 1;
    if (r) cc.set(r.atoms.CA, t * 3);
    const refs = names.map((n) => chem?.atoms[n]?.ref ?? [0, 0, 0]);
    // Upstream augments reference conformers per residue, except the last UID.
    const center = [0, 1, 2].map(
        (d) => refs.reduce((sum, x) => sum + x[d], 0) / refs.length,
      ),
      q = Array.from({ length: 4 }, rng.normal),
      qn = Math.hypot(...q),
      [w, x, y, z] = q.map((v) => v / qn);
    const R = [
        1 - 2 * (y * y + z * z),
        2 * (x * y - z * w),
        2 * (x * z + y * w),
        2 * (x * y + z * w),
        1 - 2 * (x * x + z * z),
        2 * (y * z - x * w),
        2 * (x * z - y * w),
        2 * (y * z + x * w),
        1 - 2 * (x * x + y * y),
      ],
      trans = Array.from({ length: 3 }, rng.normal);
    for (let j = 0; j < names.length; j++, index++) {
      const n = names[j],
        c = chem?.atoms[n];
      atomMask[index] = resolved[index] = 1;
      uid[index] = t;
      mapping[index * N + t] = 1;
      if (j < 4) bbmap[(t * 4 + j) * A + index] = 1;
      const el = designed ? (j < 4 ? [7, 6, 6, 8][j] : 114) : c.element;
      element[index * 128 + el] = 1;
      charge[index] = c?.charge ?? 0;
      for (let k = 0; k < 4; k++)
        chars[(index * 4 + k) * 64 + (n[k] ? n.charCodeAt(k) - 32 : 0)] = 1;
      if (r) {
        coords.set(r.atoms[n], index * 3);
        let ref = refs[j];
        if (t < N - 1)
          ref = [0, 1, 2].map(
            (d) =>
              trans[d] +
              [0, 1, 2].reduce(
                (s, k) => s + (refs[j][k] - center[k]) * R[k * 3 + d],
                0,
              ),
          );
        pos.set(ref, index * 3);
      }
      atoms.push({
        token: t,
        name: n,
        residue: designed ? "UNK" : r.name,
        chain:
          designed || mode === "motif" ? "A" : String.fromCharCode(65 + chain),
        sourceChain: r?.chain,
        residueId: designed || mode === "motif" ? String(t + 1) : r.id,
        designed,
      });
    }
  }
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++) if (slots[i] && slots[j]) td[i * N + j] = 1;
  const center = [0, 1, 2].map(
    (d) => coords.reduce((sum, v, i) => sum + (i % 3 === d ? v : 0), 0) / count,
  );
  for (let i = 0; i < count; i++)
    for (let d = 0; d < 3; d++) coords[i * 3 + d] -= center[d];
  f.metadata = {
    L,
    nDesigned: slots.filter((r) => !r).length,
    nTokens: N,
    nAtoms: count,
    mode,
    atoms,
  };
  return f;
}
// Geometric input features contain no learned values and are independent of precision.
export function derivedFeatures(features) {
  const get = (k) => features.get("f:" + k),
    d = (k) => get(k).data,
    N = get("token_pad_mask").shape[1],
    A = get("atom_pad_mask").shape[1];
  const out = {},
    put = (k, shape) =>
      (out[k] = {
        shape,
        data: new Float32Array(shape.reduce((a, b) => a * b, 1)),
      });
  const rel = put("relative_features", [1, N, N, 139]).data,
    dist = put("distance_bins", [1, N, N, 38]).data,
    bbdist = put("bb_dist", [1, 4, N, N, 1]).data;
  const cc = d("center_coords"),
    r = d("coords"),
    bbmap = d("token_to_bb4_atoms"),
    f = Math.fround;
  const br = new Float32Array(N * 4 * 3);
  for (let i = 0; i < N * 4; i++)
    for (let a = 0; a < A; a++)
      if (bbmap[i * A + a])
        for (let k = 0; k < 3; k++)
          br[i * 3 + k] += bbmap[i * A + a] * r[a * 3 + k];
  const norm = (a, b, ia, ib) =>
    Math.sqrt(
      [0, 1, 2].reduce((s, k) => s + f(f(a[ia + k] - b[ib + k]) ** 2), 0),
    );
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++) {
      const idx = i * N + j,
        base = idx * 139,
        same = d("feature_asym_id")[i] === d("feature_asym_id")[j],
        ent = d("entity_id")[i] === d("entity_id")[j],
        ri = d("feature_residue_index")[i],
        rj = d("feature_residue_index")[j];
      rel[base + (same ? Math.max(0, Math.min(64, ri - rj + 32)) : 65)] = 1;
      rel[
        base +
          66 +
          (same && ri === rj
            ? Math.max(
                0,
                Math.min(64, d("token_index")[i] - d("token_index")[j] + 32),
              )
            : 65)
      ] = 1;
      rel[base + 132] = ent ? 1 : 0;
      rel[
        base +
          133 +
          (ent
            ? Math.max(0, Math.min(4, d("sym_id")[i] - d("sym_id")[j] + 2))
            : 5)
      ] = 1;
      const distance = norm(cc, cc, i * 3, j * 3);
      let bin = 0;
      while (bin < 37 && distance > f(3.25 + (bin * 47.5) / 36)) bin++;
      dist[idx * 38 + bin] = 1;
      for (let a = 0; a < 4; a++)
        bbdist[(a * N + i) * N + j] = norm(
          br,
          br,
          (i * 4 + a) * 3,
          (j * 4 + a) * 3,
        );
    }
  const T = get("template_mask").shape[1],
    tf = put("template_features", [1, T, N, N, 109]).data;
  const cb = d("template_cb"),
    ca = d("template_ca"),
    rot = d("template_frame_rot"),
    translation = d("template_frame_t"),
    restype = d("template_restype"),
    fm = d("template_mask_frame"),
    cm = d("template_mask_cb"),
    vis = d("visibility_ids");
  for (let t = 0; t < T; t++)
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++) {
        const ai = t * N + i,
          aj = t * N + j,
          b = ((t * N + i) * N + j) * 109,
          visible = vis[ai] === vis[aj];
        let bin = 0;
        const distance = norm(cb, cb, ai * 3, aj * 3);
        while (bin < 37 && distance > f(3.25 + (bin * 47.5) / 36)) bin++;
        if (visible) {
          tf[b + bin] = 1;
          tf[b + 38] = cm[ai] * cm[aj];
          tf[b + 42] = fm[ai] * fm[aj];
          for (let k = 0; k < 3; k++) {
            let v = 0;
            for (let m = 0; m < 3; m++)
              v +=
                rot[aj * 9 + m * 3 + k] *
                (ca[ai * 3 + m] - translation[aj * 3 + m]);
            tf[b + 39 + k] = Math.sign(v);
          }
        }
        tf.set(restype.subarray(ai * 33, ai * 33 + 33), b + 43);
        tf.set(restype.subarray(aj * 33, aj * 33 + 33), b + 76);
      }
  return out;
}
export function validateFeatures(features) {
  if (!(features instanceof Map))
    throw Error("features must be a Map of named tensors");
  const get = (k) => {
    const t = features.get("f:" + k);
    if (!t) throw Error(`Missing feature ${k}`);
    return t;
  };
  const N = get("token_pad_mask").shape[1],
    A = get("atom_pad_mask").shape[1],
    T = get("template_mask").shape[1],
    S = get("msa").shape[1];
  if (
    !Number.isSafeInteger(N) ||
    N < 1 ||
    !Number.isSafeInteger(A) ||
    A < 1 ||
    A % 32 ||
    !Number.isSafeInteger(T) ||
    T < 1 ||
    !Number.isSafeInteger(S) ||
    S < 1
  )
    throw Error("Invalid token, atom-window, template, or MSA dimensions");
  const shapes = {
    ref_pos: [1, A, 3],
    ref_charge: [1, A],
    ref_element: [1, A, 128],
    ref_atom_name_chars: [1, A, 4, 64],
    ref_space_uid: [1, A],
    atom_pad_mask: [1, A],
    atom_to_token: [1, A, N],
    token_pad_mask: [1, N],
    res_type: [1, N, 33],
    profile: [1, N, 33],
    deletion_mean: [1, N],
    token_to_bb4_atoms: [1, N, 4, A],
    coords: [1, 1, A, 3],
    center_coords: [1, N, 3],
    token_bonds: [1, N, N, 1],
    type_bonds: [1, N, N],
    contact_conditioning: [1, N, N, 5],
    contact_threshold: [1, N, N],
    token_distance_mask: [1, N, N],
  };
  for (const k of [
    "feature_asym_id",
    "feature_residue_index",
    "entity_id",
    "token_index",
    "sym_id",
    "cyclic",
    "method_feature",
    "modified",
    "mol_type",
    "design_mask",
    "binding_type",
    "ss_type",
  ])
    shapes[k] = [1, N];
  for (const k of [
    "msa",
    "msa_mask",
    "msa_paired",
    "has_deletion",
    "deletion_value",
  ])
    shapes[k] = [1, S, N];
  for (const k of [
    "template_mask",
    "template_mask_cb",
    "template_mask_frame",
    "visibility_ids",
  ])
    shapes[k] = [1, T, N];
  for (const k of ["template_ca", "template_cb", "template_frame_t"])
    shapes[k] = [1, T, N, 3];
  shapes.template_restype = [1, T, N, 33];
  shapes.template_frame_rot = [1, T, N, 3, 3];
  for (const [k, shape] of Object.entries(shapes)) {
    const t = get(k);
    if (
      String(t.shape) !== String(shape) ||
      t.data.length !== shape.reduce((a, b) => a * b, 1) ||
      !t.data.every(Number.isFinite)
    )
      throw Error(`Invalid feature ${k}; expected [${shape}] finite values`);
  }
  if (!get("atom_pad_mask").data.some((v) => v === 1))
    throw Error("Input has no atoms");
  return { N, A };
}
