import { linearAssignment } from './assignment.js';
import { cyclicSymmetryFeatures } from './symmetry.js';
import { parsePDB, parseMMCIF, parsePDBConect, parseMMCIFBonds, writePDB, writeMMCIF } from './structure.js';
import { chemistry } from './chemistry.js';
import { sequenceEntropy } from './sampler.js';

const unknownNames = ['N', 'CA', 'C', 'O', 'CB', ...Array.from({ length: 9 }, (_, i) => 'V' + i)];
const elements = 'X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'.toUpperCase().split(' ');
const elementSet = new Set(elements.slice(1));
const covalentRadii = {
  H: 0.31, B: 0.84, C: 0.76, N: 0.71, O: 0.66, F: 0.57, NA: 1.66, MG: 1.41, AL: 1.21,
  SI: 1.11, P: 1.07, S: 1.05, CL: 1.02, K: 2.03, CA: 1.76, MN: 1.39, FE: 1.32, CO: 1.26,
  NI: 1.24, CU: 1.32, ZN: 1.22, SE: 1.20, BR: 1.20, MO: 1.54, I: 1.39, PT: 1.36, AU: 1.36,
};
const DEFAULT_CFG_FEATURES = ['active_donor', 'active_acceptor', 'ref_atomwise_rasa'];
const TOKEN_FEATURE_KEYS = new Set([
  'asym_id', 'entity_id', 'sym_id', 'is_dna', 'is_ligand', 'is_motif_token_unindexed',
  'is_motif_token_with_fully_fixed_coord', 'is_non_loopy', 'is_polar', 'is_rna',
  'residue_index', 'token_index', 'is_protein', 'ref_plddt', 'restype',
  'ref_motif_token_type', 'terminus_type',
]);
const TOKEN_2D_FEATURE_KEYS = new Set(['token_bonds', 'unindexing_pair_mask']);
const ATOM_FEATURE_KEYS = new Set([
  'active_acceptor', 'active_donor', 'has_zero_occupancy', 'is_atom_level_hotspot',
  'is_motif_atom_unindexed', 'is_motif_atom_with_fixed_coord', 'is_motif_atom_with_fixed_seq',
  'ref_charge', 'ref_is_motif_atom_unindexed', 'ref_is_motif_atom_with_fixed_coord',
  'ref_mask', 'ref_pos_is_ground_truth', 'atom_to_token_map', 'ref_space_uid',
  'is_backbone', 'is_ca', 'is_central', 'is_sidechain', 'is_virtual',
  'motif_pos', 'ref_atomwise_rasa', 'ref_pos', 'ref_atom_name_chars', 'ref_element',
  'partial_t',
]);
const protein = r => Boolean(chemistry.atom14[r]);
const dna = r => ['DA', 'DC', 'DG', 'DT', 'DN'].includes(r);
const rna = r => ['A', 'C', 'G', 'U', 'N'].includes(r);
const polar = new Set(['SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS', 'HIS', 'LYS', 'ARG', 'ASP', 'GLU']);

function residueMatch(selection, key) {
  if (selection === key) return true;
  const k = /^([^:]+):(-?\d+)$/.exec(key) ?? /^([A-Za-z0-9]+?)(-?\d+)$/.exec(key);
  const m = /^([^:]+):(-?\d+)(?:-(-?\d+))?$/.exec(selection) ?? /^([A-Za-z0-9]+?)(-?\d+)(?:-(-?\d+))?$/.exec(selection);
  return Boolean(m && k && m[1] === k[1] && +k[2] >= +m[2] && +k[2] <= +(m[3] ?? m[2]));
}

function selected(selection, key, atom, defaultValue = false) {
  if (selection == null) return defaultValue;
  if (typeof selection === 'boolean') return selection;
  if (typeof selection === 'string') return selection.split(',').some(s => residueMatch(s.trim(), key));
  if (typeof selection !== 'object' || Array.isArray(selection))
    throw new Error('Atom selection must be boolean, residue string, or residue-to-atoms map');
  return Object.entries(selection).some(([s, value]) =>
    residueMatch(s, key) && (value === true || value === 'ALL' ||
      String(value).split(',').some(v => v.trim() === atom || (v.trim() === 'BKBN' && ['N', 'CA', 'C', 'O'].includes(atom)))));
}

function inferCovalentBonds(atoms) {
  const bonds = [];
  if (atoms.length <= 1) return bonds;
  for (let i = 0; i < atoms.length; i++) {
    const ai = atoms[i], ri = covalentRadii[ai.element?.toUpperCase()] ?? 0.77;
    for (let j = i + 1; j < atoms.length; j++) {
      const aj = atoms[j], rj = covalentRadii[aj.element?.toUpperCase()] ?? 0.77;
      const d = Math.hypot(
        ai.position[0] - aj.position[0],
        ai.position[1] - aj.position[1],
        ai.position[2] - aj.position[2],
      );
      if (d > 0.4 && d <= ri + rj + 0.45) bonds.push([i, j, 1]);
    }
  }
  return bonds;
}

function readResidues(source, format) {
  const isString = typeof source === 'string';
  const isCif = isString && (format === 'cif' || format === 'mmcif' || /^\s*data_/m.test(source));
  const rows = Array.isArray(source) ? source : source ? (isCif ? parseMMCIF(source) : parsePDB(source)) : [];
  const groups = new Map(), identities = new Set(), serialMap = new Map(), model = rows[0]?.pdbx_pdb_model_num;
  for (const row of rows) {
    if (row.pdbx_pdb_model_num !== model) throw new Error('Select a single input model');
    if (!['.', '?', '', undefined].includes(row.pdbx_pdb_ins_code)) throw new Error('Insertion codes require explicit atom records');
    if (!['.', '?', '', undefined, 'A'].includes(row.label_alt_id)) continue;
    const chainId = row.auth_asym_id ?? row.label_asym_id, residueId = row.auth_seq_id ?? row.label_seq_id;
    if (!/^-?\d+$/.test(String(residueId))) throw new Error('Input residues need integer author or label sequence IDs');
    const key = chainId + ':' + residueId, name = row.label_atom_id, identity = key + ':' + name;
    if (identities.has(identity)) throw new Error('Duplicate input atom ' + identity);
    identities.add(identity);
    if (['H', 'D'].includes(row.type_symbol?.toUpperCase()) || row.label_atom_id === 'OXT') continue;
    if (!groups.has(key)) {
      groups.set(key, {
        key, displayKey: chainId + residueId, sourceChain: chainId, sourceResId: Number(residueId),
        resname: row.label_comp_id, atoms: [], bonds: undefined,
      });
    }
    const rawSym = row.type_symbol && !['?', '.'].includes(row.type_symbol) ? row.type_symbol.toUpperCase() : null;
    const compUpper = String(row.label_comp_id ?? '').toUpperCase();
    const element = rawSym ??
      (protein(row.label_comp_id) ? name[0] :
        chemistry.references[row.label_comp_id]?.atoms.find(a => a.name === name)?.element ??
        (elementSet.has(compUpper) && name.toUpperCase() === compUpper ? compUpper : undefined));
    if (!element) throw new Error('Missing element for ' + identity);
    const group = groups.get(key);
    const localIdx = group.atoms.length;
    group.atoms.push({
      name, element,
      position: ['cartn_x', 'cartn_y', 'cartn_z'].map(k => Number(String(row[k]).replace(/\(\d+\)/, ''))),
      charge: Number(row.pdbx_formal_charge) || 0,
    });
    if (row.id != null && row.id !== '' && row.id !== '?') serialMap.set(String(row.id).trim(), { key, index: localIdx });
  }
  if (isString && !isCif) {
    const conect = parsePDBConect(source);
    if (conect.length) {
      const seen = new Map();
      for (const [aId, bId] of conect) {
        const a = serialMap.get(aId), b = serialMap.get(bId);
        if (!a || !b || a.key !== b.key || a.index === b.index) continue;
        const lo = Math.min(a.index, b.index), hi = Math.max(a.index, b.index);
        if (!seen.has(a.key)) seen.set(a.key, new Set());
        const edgeKey = lo + ':' + hi;
        if (seen.get(a.key).has(edgeKey)) continue;
        seen.get(a.key).add(edgeKey);
        const g = groups.get(a.key);
        if (!g.bonds) g.bonds = [];
        g.bonds.push([lo, hi, 1]);
      }
    }
  } else if (isString && isCif) {
    const cifBonds = parseMMCIFBonds(source);
    if (cifBonds.size) {
      for (const group of groups.values()) {
        const pairs = cifBonds.get(group.resname);
        if (!pairs?.length) continue;
        const idxByName = new Map(group.atoms.map((a, idx) => [a.name, idx]));
        const seen = new Set();
        group.bonds = [];
        for (const [n1, n2] of pairs) {
          const i = idxByName.get(n1), j = idxByName.get(n2);
          if (i === undefined || j === undefined || i === j) continue;
          const lo = Math.min(i, j), hi = Math.max(i, j), edge = lo + ':' + hi;
          if (seen.has(edge)) continue;
          seen.add(edge);
          group.bonds.push([lo, hi, 1]);
        }
      }
    }
  }
  return groups;
}

export function buildReferenceConformer(name, { atoms, bonds } = {}) {
  const known = chemistry.references[name];
  if (known) return { atoms: known.atoms.map(a => ({ ...a, position: a.position.slice() })), bonds: known.bonds.map(b => b.slice()) };
  if (atoms) {
    const resolvedBonds = bonds ?? (atoms.length <= 1 ? [] : inferCovalentBonds(atoms));
    return {
      atoms: atoms.map(a => ({ ...a, position: a.position.slice() })),
      bonds: resolvedBonds.map(b => b.slice()),
    };
  }
  throw new Error('No reference chemistry for ' + name + '; supply atoms and bonds explicitly');
}

function defaultTargetContig(source) {
  const parts = [];
  let prevChain = null;
  for (const [key, r] of source) {
    if (!protein(r.resname)) continue;
    if (prevChain !== null && r.sourceChain !== prevChain) parts.push('/0');
    parts.push(key);
    prevChain = r.sourceChain;
  }
  return parts.join(',');
}

function inferOriginFromHotspots(residues, hotspotSelection) {
  const hotspotAtoms = [];
  const finiteAtoms = [];
  for (const r of residues) {
    if (!r.key) continue;
    for (const a of r.atoms) {
      if (a.padded || a.element === 'VX' || !a.position.every(Number.isFinite)) continue;
      if (a.fixed) finiteAtoms.push(a);
      if (a.fixed && selected(hotspotSelection, r.key, a.sourceName ?? a.name)) hotspotAtoms.push(a);
    }
  }
  if (!hotspotAtoms.length) return null;
  const hotspotCom = [0, 0, 0];
  for (const a of hotspotAtoms) for (let c = 0; c < 3; c++) hotspotCom[c] += a.position[c];
  for (let c = 0; c < 3; c++) hotspotCom[c] /= hotspotAtoms.length;
  const nearby = finiteAtoms.filter(a => hotspotAtoms.some(h =>
    Math.hypot(a.position[0] - h.position[0], a.position[1] - h.position[1], a.position[2] - h.position[2]) <= 12.0));
  const core = nearby.length ? nearby : finiteAtoms;
  const coreCom = [0, 0, 0];
  for (const a of core) for (let c = 0; c < 3; c++) coreCom[c] += a.position[c];
  for (let c = 0; c < 3; c++) coreCom[c] /= core.length;
  const vec = [hotspotCom[0] - coreCom[0], hotspotCom[1] - coreCom[1], hotspotCom[2] - coreCom[2]];
  const norm = Math.hypot(vec[0], vec[1], vec[2]);
  if (norm <= 1e-6) return hotspotCom.map(Math.fround);
  return [0, 1, 2].map(c => Math.fround(hotspotCom[c] + 10.0 * (vec[c] / norm)));
}

export function buildCfgReferenceFeatures(features, cfgFeatures = DEFAULT_CFG_FEATURES) {
  const n = features.residue_index.length;
  const l = features.atom_to_token_map.length;
  const firstUnindexedToken = features.is_motif_token_unindexed.findIndex(v => v !== 0);
  const firstUnindexedAtom = features.is_motif_atom_unindexed.findIndex(v => v !== 0);
  const nCrop = firstUnindexedToken >= 0 ? firstUnindexedToken : n;
  const lCrop = firstUnindexedAtom >= 0 ? firstUnindexedAtom : l;
  const zeroKeys = new Set(cfgFeatures ?? DEFAULT_CFG_FEATURES);
  const stripped = {};
  for (const [key, value] of Object.entries(features)) {
    if (!(value instanceof Float32Array)) {
      stripped[key] = value;
      continue;
    }
    let out;
    if (TOKEN_2D_FEATURE_KEYS.has(key)) {
      out = new Float32Array(nCrop * nCrop);
      for (let i = 0; i < nCrop; i++) {
        for (let j = 0; j < nCrop; j++) out[i * nCrop + j] = value[i * n + j];
      }
    } else if (TOKEN_FEATURE_KEYS.has(key)) {
      const stride = value.length / n;
      out = value.slice(0, nCrop * stride);
    } else if (ATOM_FEATURE_KEYS.has(key)) {
      const stride = value.length / l;
      out = value.slice(0, lCrop * stride);
    } else {
      out = value.slice();
    }
    if (zeroKeys.has(key)) out.fill(0);
    stripped[key] = out;
  }
  return stripped;
}

// Deterministic, browser-native inference featurizer. Lengths are concrete, not sampled ranges.
export function buildInput(options = {}) {
  if (options.symmetry?.id) {
    const m = /^C([2-9]|[1-9][0-9]+)$/.exec(options.symmetry.id);
    if (!m || options.structure || options.input || options.components || options.contig || options.unindex)
      throw new Error('High-level symmetry supports de novo Cn; use explicit features for other frames');
    const order = +m[1], built = buildInput({ ...options, symmetry: undefined, contig: Array(order).fill(String(options.length)).join(',/0,') });
    built.features = cyclicSymmetryFeatures(built.features, order);
    if ((options.cfgScale ?? 1) !== 1)
      built.referenceFeatures = options.referenceFeatures ?? buildCfgReferenceFeatures(built.features, options.cfgFeatures ?? options.cfg_features);
    return built;
  }
  const mode = options.mode ?? 'unconditional';
  if (!['unconditional', 'motif', 'binder', 'conditioned'].includes(mode)) throw new Error('Unknown generation mode');
  const partialT = options.partialT ?? options.partial_t;
  const source = readResidues(options.structure ?? options.input, options.format), residues = [];
  let chain = 0, residue = 0;
  const addDeNovo = count => {
    if (!Number.isInteger(count) || count < 1) throw new Error('Use a positive concrete length');
    for (let i = 0; i < count; i++) residues.push({
      resname: 'MSK', key: null, chain, residue: residue++,
      atoms: unknownNames.map((name, j) => ({ name, element: j < 5 ? ['N', 'C', 'C', 'O', 'C'][j] : 'VX', position: [0, 0, 0] })),
    });
  };
  const addSource = (key, unindexed = false, unindexGroup = 0) => {
    const original = source.get(key);
    if (!original) throw new Error('Missing input residue ' + key);
    const r = { ...original, chain, residue: residue++, unindexed, unindexGroup, atoms: original.atoms.map(a => ({ ...a, position: a.position.slice() })) };
    if (unindexed) r.atoms = r.atoms.filter(a => selected(options.select_fixed_atoms ?? options.fixedAtoms, key, a.name, true));
    if (!r.atoms.length) throw new Error('Empty atom selection ' + key);
    if (protein(r.resname) && !unindexed) {
      const order = chemistry.atom14[r.resname];
      r.atoms.sort((a, b) => order.indexOf(a.name) - order.indexOf(b.name));
      if (r.atoms.some(a => !order.includes(a.name))) throw new Error('Unsupported protein atom in ' + key);
    }
    if (r.protein ?? protein(r.resname)) {
      const flags = r.atoms.map(a => selected(options.select_unfixed_sequence ?? options.unfixedSequence, key, a.name));
      if (flags.some(Boolean) && !flags.every(Boolean)) throw new Error('Sequence conditioning must select whole protein residues');
      if (flags.every(Boolean) && !unindexed) {
        r.redesign = true;
        const origMap = new Map(r.atoms.map(a => [a.name, a]));
        const padPos = (origMap.get('CB') ?? origMap.get('CA'))?.position ?? [0, 0, 0];
        r.atoms = unknownNames.map((name, i) => {
          const actual = origMap.get(chemistry.atom14[r.resname][i]);
          return {
            ...(actual ?? {}), name, sourceName: actual?.name, padded: !actual,
            element: i < 5 ? ['N', 'C', 'C', 'O', 'C'][i] : 'VX',
            position: actual ? actual.position.slice() : (partialT !== undefined ? padPos.slice() : [0, 0, 0]),
          };
        });
      }
    }
    residues.push(r);
  };
  const expand = contig => {
    for (const part of String(contig).split(',').map(s => s.trim())) {
      if (part === '/0') { chain++; residue = 0; continue; }
      if (/^\d+$/.test(part)) { addDeNovo(+part); continue; }
      const matches = [...source.keys()].filter(k => residueMatch(part, k));
      if (!matches.length) throw new Error('Unresolved contig component ' + part);
      for (const key of matches) addSource(key);
    }
  };
  const contig = options.contig ?? (mode === 'binder' ? `${options.length},/0,${options.target ?? defaultTargetContig(source)}` : null);
  if (contig) expand(contig); else addDeNovo(Number(options.length));
  if (options.unindex) {
    const groups = typeof options.unindex === 'string' ? options.unindex.split(',') : Object.keys(options.unindex);
    for (const [group, selection] of groups.entries())
      for (const key of source.keys()) if (residueMatch(selection.trim(), key)) addSource(key, true, group + 1);
  }
  if (options.ligand) {
    chain++; residue = 0;
    const names = String(options.ligand).split(',').map(s => s.trim());
    for (const [key, r] of source)
      if (names.includes(r.resname) || names.includes(key) || names.includes(r.displayKey)) addSource(key);
  }
  for (const component of options.components ?? []) {
    chain++; residue = 0;
    const reference = buildReferenceConformer(component.name, component);
    residues.push({
      resname: component.name, key: component.id ?? 'component' + chain, chain, residue: 0,
      atoms: reference.atoms.map(a => ({ ...a, position: a.position.map((x, c) => x + (component.translation?.[c] ?? 0)) })),
      bonds: reference.bonds,
    });
  }
  if (options.cyclic && (residues.some(r => r.key) || chain !== 0)) throw new Error('Cyclic encoding requires one complete de novo peptide');
  if (chain > 25) throw new Error('High-level PDB export supports up to 26 chains');
  const tokens = [];
  for (let uid = 0; uid < residues.length; uid++) {
    const r = residues[uid];
    r.uid = uid; r.protein = protein(r.resname); r.dna = dna(r.resname); r.rna = rna(r.resname); r.ligand = !r.protein && !r.dna && !r.rna;
    for (const a of r.atoms) {
      a.fixed = Boolean(r.key) && selected(options.select_fixed_atoms ?? options.fixedAtoms, r.key, a.sourceName ?? a.name, true) && (!r.redesign || Boolean(a.sourceName));
      a.seq = Boolean(r.key) && !r.redesign && (!r.protein || !selected(options.select_unfixed_sequence ?? options.unfixedSequence, r.key, a.name));
    }
    if (r.ligand) for (const atom of r.atoms) tokens.push({ ...r, atoms: [atom] });
    else tokens.push(r);
  }
  const n = tokens.length, l = tokens.reduce((s, r) => s + r.atoms.length, 0);
  const maxBytes = options.maxStorageBufferBindingSize ?? 2 ** 31;
  if (n < 1 || l < 1 || l * l * 16 * 4 > maxBytes || n * n * 128 * 4 > maxBytes)
    throw new RangeError('Input exceeds dense pair activation limit');
  const f = {}, zeros = (key, size) => f[key] = new Float32Array(size);
  for (const key of ['active_acceptor', 'active_donor', 'has_zero_occupancy', 'is_atom_level_hotspot', 'is_motif_atom_unindexed', 'is_motif_atom_with_fixed_coord', 'is_motif_atom_with_fixed_seq', 'ref_charge', 'ref_is_motif_atom_unindexed', 'ref_is_motif_atom_with_fixed_coord', 'ref_mask', 'ref_pos_is_ground_truth', 'atom_to_token_map', 'ref_space_uid', 'is_backbone', 'is_ca', 'is_central', 'is_sidechain', 'is_virtual']) zeros(key, l);
  for (const key of ['asym_id', 'entity_id', 'sym_id', 'is_dna', 'is_ligand', 'is_motif_token_unindexed', 'is_motif_token_with_fully_fixed_coord', 'is_non_loopy', 'is_polar', 'is_rna', 'residue_index', 'token_index', 'is_protein', 'ref_plddt']) zeros(key, n);
  for (const key of ['motif_pos', 'ref_atomwise_rasa', 'ref_pos']) zeros(key, l * 3);
  for (const key of ['token_bonds', 'unindexing_pair_mask']) zeros(key, n * n);
  zeros('ref_atom_name_chars', l * 256); zeros('ref_element', l * 128); zeros('restype', n * 32); zeros('ref_motif_token_type', n * 3); zeros('terminus_type', n * 2);
  const entities = new Map(), chainEntity = new Map(), chainSym = new Map(), chains = [...new Set(residues.map(r => r.chain))];
  for (const c of chains) {
    const signature = residues.filter(r => r.chain === c).map(r => r.resname).join(',');
    if (!entities.has(signature)) entities.set(signature, entities.size);
    const e = entities.get(signature);
    chainEntity.set(c, e);
    chainSym.set(c, [...chainEntity.values()].filter(v => v === e).length - 1);
  }
  if (partialT !== undefined && residues.some(r => !r.key) && !options.coordinates)
    throw new Error('Partial diffusion requires coordinates for every diffused atom');
  const oriToken = options.ori_token ?? options.oriToken;
  const hotspotSel = options.select_hotspots ?? options.hotspots;
  const inferOriStrategy = options.infer_ori_strategy ?? options.inferOriStrategy ??
    (mode === 'binder' && hotspotSel != null && partialT === undefined ? 'hotspots' : undefined);
  let origin = [0, 0, 0];
  if (oriToken != null) {
    if (!Array.isArray(oriToken) || oriToken.length !== 3 || !oriToken.every(Number.isFinite))
      throw new Error('ori_token must be a 3-element finite coordinate array');
    origin = oriToken.map(Math.fround);
  } else if (partialT !== undefined) {
    const realAtoms = residues.flatMap(r => r.atoms).filter(a => !a.padded && a.element !== 'VX');
    const diffusedAtoms = realAtoms.filter(a => !a.fixed);
    const centerAtoms = diffusedAtoms.length ? diffusedAtoms : realAtoms;
    for (const a of centerAtoms) for (let c = 0; c < 3; c++) origin[c] = Math.fround(origin[c] + Math.fround(a.position[c]));
    if (centerAtoms.length) for (let c = 0; c < 3; c++) origin[c] = Math.fround(origin[c] / centerAtoms.length);
  } else if (inferOriStrategy === 'hotspots') {
    const inferred = inferOriginFromHotspots(residues, hotspotSel);
    if (inferred) origin = inferred;
    else {
      const fixedAtoms = residues.flatMap(r => r.atoms).filter(a => a.fixed);
      for (const a of fixedAtoms) for (let c = 0; c < 3; c++) origin[c] = Math.fround(origin[c] + Math.fround(a.position[c]));
      if (fixedAtoms.length) for (let c = 0; c < 3; c++) origin[c] = Math.fround(origin[c] / fixedAtoms.length);
    }
  } else if (inferOriStrategy === 'com') {
    const realAtoms = residues.filter(r => r.key).flatMap(r => r.atoms).filter(a => !a.padded && a.element !== 'VX');
    for (const a of realAtoms) for (let c = 0; c < 3; c++) origin[c] = Math.fround(origin[c] + Math.fround(a.position[c]));
    if (realAtoms.length) for (let c = 0; c < 3; c++) origin[c] = Math.fround(origin[c] / realAtoms.length);
  } else {
    const fixedAtoms = residues.flatMap(r => r.atoms).filter(a => a.fixed);
    for (const a of fixedAtoms) for (let c = 0; c < 3; c++) origin[c] = Math.fround(origin[c] + Math.fround(a.position[c]));
    if (fixedAtoms.length) for (let c = 0; c < 3; c++) origin[c] = Math.fround(origin[c] / fixedAtoms.length);
  }
  const nonLoopy = options.isNonLoopy ?? options.is_non_loopy;
  const atoms = [];
  let index = 0;
  for (let t = 0; t < n; t++) {
    const r = tokens[t], p = r.protein, un = Boolean(r.unindexed), known = r.atoms.some(a => a.seq), motif = r.atoms.some(a => a.fixed || a.seq) || un;
    f.asym_id[t] = r.chain; f.entity_id[t] = chainEntity.get(r.chain); f.sym_id[t] = chainSym.get(r.chain);
    f.residue_index[t] = r.residue; f.token_index[t] = t;
    f.is_protein[t] = +p; f.is_dna[t] = +r.dna; f.is_rna[t] = +r.rna; f.is_ligand[t] = +r.ligand; f.is_polar[t] = +polar.has(r.resname);
    f.restype[t * 32 + (known ? (chemistry.restypes[r.resname] ?? 20) : 31)] = 1;
    f.ref_motif_token_type[t * 3 + (un ? 2 : motif ? 1 : 0)] = 1;
    f.ref_plddt[t] = !motif && options.plddt_enhanced !== false ? 1 : 0;
    f.is_motif_token_unindexed[t] = +un;
    f.is_motif_token_with_fully_fixed_coord[t] = +r.atoms.every(a => a.fixed);
    f.is_non_loopy[t] = nonLoopy == null || motif ? 0 : (nonLoopy ? 1 : -1);
    const chainProtein = residues.filter(a => a.chain === r.chain && a.protein && !a.unindexed);
    if (p && !un) { f.terminus_type[t * 2] = +(r === chainProtein.at(-1)); f.terminus_type[t * 2 + 1] = +(r === chainProtein[0]); }
    const ca = un ? r.atoms[0].name : p ? 'CA' : r.ligand ? r.atoms[0].name : ['C', 'U', 'DC', 'DT'].includes(r.resname) ? 'C2' : 'C4';
    const central = un ? r.atoms[0].name : p ? (r.resname === 'GLY' ? 'CA' : 'CB') : ca;
    if (!r.atoms.some(a => a.name === ca) || !r.atoms.some(a => a.name === central))
      throw new Error('Missing representative atom in ' + r.key);
    for (const a of r.atoms) {
      const i = index++, isVirtual = a.element === 'VX', ref = !p && !un && a.seq, element = ref ? elements.indexOf(a.element.toUpperCase()) : 0;
      if (element < 0) throw new Error('Unknown chemical element ' + a.element);
      f.atom_to_token_map[i] = t; f.ref_space_uid[i] = r.uid;
      f.is_ca[i] = +(a.name === ca); f.is_central[i] = +(a.name === central); f.is_virtual[i] = +isVirtual;
      f.is_backbone[i] = +(r.ligand || (p && ['N', 'CA', 'C', 'O'].includes(a.name)));
      f.is_sidechain[i] = +(p && !f.is_backbone[i]);
      f.is_motif_atom_with_fixed_coord[i] = f.ref_is_motif_atom_with_fixed_coord[i] = +a.fixed;
      f.is_motif_atom_with_fixed_seq[i] = +a.seq;
      f.is_motif_atom_unindexed[i] = f.ref_is_motif_atom_unindexed[i] = +un;
      f.ref_element[i * 128 + element] = 1; f.ref_mask[i] = +ref; f.ref_charge[i] = ref ? a.charge ?? 0 : 0;
      const name = ref && r.ligand ? a.element : p && r.key && !r.redesign && !un ? unknownNames[chemistry.atom14[r.resname].indexOf(a.name)] : a.name;
      if (name.length > 4) throw new Error('Atom name exceeds AF3 encoding');
      for (let c = 0; c < 4; c++) {
        const char = name.padEnd(4)[c].charCodeAt(0) - 32;
        if (char < 0 || char > 63) throw new Error('Atom names must use AF3 ASCII range');
        f.ref_atom_name_chars[i * 256 + c * 64 + char] = 1;
      }
      if (a.fixed || partialT !== undefined) for (let c = 0; c < 3; c++) f.motif_pos[i * 3 + c] = Math.fround(a.position[c]) - Math.fround(origin[c]);
      f.active_donor[i] = +selected(options.select_hbond_donor ?? options.hbondDonor, r.key, a.name);
      f.active_acceptor[i] = +selected(options.select_hbond_acceptor ?? options.hbondAcceptor, r.key, a.name);
      f.is_atom_level_hotspot[i] = +selected(hotspotSel, r.key, a.name);
      for (const [bin, key] of ['select_buried', 'select_partially_buried', 'select_exposed'].entries()) {
        if (options[key] != null) {
          f.ref_atomwise_rasa.fill(0, i * 3, i * 3 + 3);
          if (selected(options[key], r.key, a.name)) f.ref_atomwise_rasa[i * 3 + bin] = 1;
        }
      }
      atoms.push({ ...a, token: t, uid: r.uid, chain: r.chain, residue: r.residue, resname: r.resname, unindexed: un });
    }
  }
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      f.unindexing_pair_mask[i * n + j] = Number(Boolean(tokens[i].unindexed) !== Boolean(tokens[j].unindexed) || (Boolean(tokens[i].unindexed) && tokens[i].unindexGroup !== tokens[j].unindexGroup));
  for (const r of residues.filter(r => r.ligand)) {
    const ref = buildReferenceConformer(r.resname, { atoms: r.atoms, bonds: r.bonds }), byName = new Map();
    for (let i = 0; i < n; i++) if (tokens[i].uid === r.uid) byName.set(tokens[i].atoms[0].name, i);
    for (const [a, b] of ref.bonds) {
      const i = byName.get(ref.atoms[a].name), j = byName.get(ref.atoms[b].name);
      if (i !== undefined && j !== undefined) f.token_bonds[i * n + j] = f.token_bonds[j * n + i] = 1;
    }
  }
  if (options.cyclic) {
    if (n < 3) throw new Error('Cyclic peptide requires at least three residues');
    f.cyclic_asym_ids = new Float32Array([0]);
  }
  if (partialT !== undefined) f.partial_t = new Float32Array(l).fill(partialT);
  const built = { features: f, atoms, tokens, origin, options };
  if ((options.cfgScale ?? 1) !== 1)
    built.referenceFeatures = options.referenceFeatures ?? buildCfgReferenceFeatures(f, options.cfgFeatures ?? options.cfg_features);
  return built;
}

function decodeProteinResname(tokenIndex, indices, logits, byIndex) {
  const predicted = byIndex[indices[tokenIndex]];
  if (chemistry.atom14[predicted] && predicted !== 'MSK') return predicted;
  if (logits && logits.length >= (tokenIndex + 1) * 32) {
    let bestClass = -1, bestVal = -Infinity;
    for (let c = 0; c < 20; c++) {
      const v = logits[tokenIndex * 32 + c];
      if (Number.isFinite(v) && v > bestVal) { bestVal = v; bestClass = c; }
    }
    if (bestClass >= 0 && chemistry.atom14[byIndex[bestClass]]) return byIndex[bestClass];
  }
  return 'UNK';
}

export function structureFromResult(input, result, processed) {
  const { atoms, tokens, origin } = input, coords = result.X_L.data, indices = result.sequence_indices_I.data;
  const logits = result.sequence_logits_I?.data;
  // Match upstream process_unindexed_outputs: assign guideposts against raw Atom14 slots BEFORE VX cleanup.
  if (!processed && tokens.some(t => t.unindexed) && input.options.cleanupGuideposts !== false) {
    const used = new Set(), restypes = {}, assignments = [], updated = coords.slice();
    const slotName = atom => {
      const token = tokens[atom.token];
      if (token.protein && token.unindexed) {
        const slot = chemistry.atom14[token.resname]?.indexOf(atom.name) ?? -1;
        if (slot >= 0) return unknownNames[slot];
      }
      return atom.name;
    };
    const atomDist = (ai, bi) => Math.hypot(
      coords[ai * 3] - coords[bi * 3],
      coords[ai * 3 + 1] - coords[bi * 3 + 1],
      coords[ai * 3 + 2] - coords[bi * 3 + 2],
    );
    for (let t = 0; t < tokens.length; t++) {
      const guide = tokens[t];
      if (!guide.unindexed) continue;
      const guideIdx = [];
      for (let i = 0; i < atoms.length; i++) if (atoms[i].token === t) guideIdx.push(i);
      const guideSlots = guideIdx.map(i => slotName(atoms[i]));
      const candIdx = [];
      for (let i = 0; i < atoms.length; i++) {
        const tok = tokens[atoms[i].token];
        if (!tok.unindexed && !tok.atoms.some(a => a.seq) && !used.has(atoms[i].token)) candIdx.push(i);
      }
      const cost = (gi, cj) => guideSlots[gi] === atoms[cj].name ? atomDist(guideIdx[gi], cj) : Infinity;
      const global = linearAssignment(guideIdx.map((_, gi) => candIdx.map(cj => cost(gi, cj))));
      const counts = new Map(), distSums = new Map();
      for (let gi = 0; gi < global.length; gi++) {
        const cj = candIdx[global[gi]], targetTok = atoms[cj].token;
        counts.set(targetTok, (counts.get(targetTok) ?? 0) + 1);
        distSums.set(targetTok, (distSums.get(targetTok) ?? 0) + cost(gi, cj));
      }
      const target = [...counts].sort((a, b) => b[1] - a[1] || distSums.get(a[0]) - distSums.get(b[0]))[0][0];
      const targetIdx = candIdx.filter(i => atoms[i].token === target);
      const local = linearAssignment(guideIdx.map((_, gi) => targetIdx.map(tj => cost(gi, tj))));
      const rmsd = Math.sqrt(guideIdx.reduce((sum, _, gi) => sum + cost(gi, targetIdx[local[gi]]) ** 2, 0) / guideIdx.length);
      used.add(target);
      restypes[target] = guide.resname;
      for (let gi = 0; gi < guideIdx.length; gi++) {
        const src = guideIdx[gi], dst = targetIdx[local[gi]];
        for (let c = 0; c < 3; c++) updated[dst * 3 + c] = coords[src * 3 + c];
      }
      assignments.push({
        source: guide.displayKey ?? guide.key, token: target,
        chain: String.fromCharCode(65 + tokens[target].chain),
        residue: tokens[target].residue + 1, insertionRmsd: rmsd,
      });
    }
    return { ...structureFromResult(input, { ...result, X_L: { ...result.X_L, data: updated } }, { restypes }), motifAssignments: assignments };
  }

  const ent = result.sequence_entropy_traj?.length
    ? result.sequence_entropy_traj.at(-1)
    : (logits?.length === tokens.length * 32 ? sequenceEntropy(logits) : null);
  const byIndex = Object.fromEntries(Object.entries(chemistry.restypes).map(([r, i]) => [i, r]));
  const rows = [];
  for (let i = 0; i < atoms.length; i++) {
    const atom = atoms[i], token = tokens[atom.token];
    if (processed && token.unindexed) continue;
    let name = atom.name, resname = atom.resname, element = atom.element;
    if (token.protein && !atom.seq && !token.unindexed) {
      resname = processed?.restypes[atom.token] ?? decodeProteinResname(atom.token, indices, logits, byIndex);
      name = chemistry.atom14[resname][unknownNames.indexOf(atom.name)];
      if (!name) continue;
      element = name[0] === 'S' ? 'S' : name[0] === 'N' ? 'N' : name[0] === 'O' ? 'O' : 'C';
    }
    if (element === 'VX') continue;
    const xyz = [0, 1, 2].map(c => coords[i * 3 + c] + origin[c]);
    const bFactor = ent && Number.isFinite(ent[atom.token]) ? ent[atom.token].toFixed(2) : '0';
    rows.push({
      group_pdb: token.ligand ? 'HETATM' : 'ATOM', id: String(rows.length + 1), type_symbol: element,
      label_atom_id: name, auth_atom_id: name, label_alt_id: '.', label_comp_id: resname, auth_comp_id: resname,
      label_asym_id: String.fromCharCode(65 + atom.chain), auth_asym_id: String.fromCharCode(65 + atom.chain),
      label_entity_id: String(input.features.entity_id[atom.token] + 1),
      label_seq_id: token.ligand ? '?' : String(atom.residue + 1), auth_seq_id: String(atom.residue + 1),
      pdbx_pdb_ins_code: '?', cartn_x: String(xyz[0]), cartn_y: String(xyz[1]), cartn_z: String(xyz[2]),
      occupancy: '1', b_iso_or_equiv: bFactor, pdbx_pdb_model_num: '1', pdbx_formal_charge: String(atom.charge ?? 0),
    });
  }
  return { rows, pdb: writePDB(rows), mmcif: writeMMCIF(rows) };
}

export async function generate(engine, options) {
  const input = buildInput(options);
  const partialT = options.partialT ?? options.partial_t;
  const result = await engine.sample({
    ...options,
    partialT,
    coordinates: options.coordinates ?? (partialT !== undefined ? input.features.motif_pos : undefined),
    features: input.features,
    referenceFeatures: options.referenceFeatures ?? input.referenceFeatures,
  });
  return { ...result, ...structureFromResult(input, result), features: input.features, origin: input.origin };
}
