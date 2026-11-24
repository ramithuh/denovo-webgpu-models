// Pure-JS, zero-dependency PDB parser, motif conditioning extractor, and Kabsch motif RMSD evaluator.
// Bit-faithful transcription of proteinfoundation/utils/motif_utils.py and align_utils.py.
import motifConstants from './motif-constants.js';
import { randomNormal, validateArray } from './sampling.js';

const ALPHABET = 'ABCDEFGHJKLMNOPQRSTUVWXYZ';
const RESTYPE_NUM = 20;

export function parsePDB(pdbText) {
  if (typeof pdbText !== 'string' || !pdbText.includes('ATOM')) {
    throw Error('Invalid PDB text: expected ATOM records');
  }
  const residues = [];
  const map = new Map();
  let sawModel = false;
  let inModel = false;
  for (const line of pdbText.split(/\r?\n/)) {
    if (line.startsWith('MODEL')) {
      if (sawModel) break;
      sawModel = true;
      inModel = true;
      continue;
    }
    if (line.startsWith('ENDMDL') && inModel) break;
    if (!line.startsWith('ATOM  ')) continue;
    const altloc = line[16] ?? ' ';
    if (altloc !== ' ' && altloc !== 'A' && altloc !== '1') continue;
    const insCode = line[26] ?? ' ';
    if (insCode !== ' ') throw Error(`PDB insertion codes are unsupported: ${line}`);
    const atomName = line.slice(12, 16).trim();
    const resName = line.slice(17, 20).trim();
    const chainId = line.slice(21, 22).trim();
    const resId = Number.parseInt(line.slice(22, 26).trim(), 10);
    const x = Number.parseFloat(line.slice(30, 38));
    const y = Number.parseFloat(line.slice(38, 46));
    const z = Number.parseFloat(line.slice(46, 54));
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      throw Error(`Nonfinite PDB coordinate: ${line}`);
    }
    const key = `${chainId}:${resId}`;
    let res = map.get(key);
    if (!res) {
      const one = motifConstants.restype_3to1[resName] ?? 'UNK';
      const resType = motifConstants.restype_order[one] ?? RESTYPE_NUM;
      res = {
        chainId,
        resId,
        resName,
        resType,
        coordsAng: new Float32Array(37 * 3),
        coordMask: new Uint8Array(37),
        availableAtoms: [],
      };
      map.set(key, res);
      residues.push(res);
    }
    const aIdx = motifConstants.atom_order[atomName];
    if (aIdx !== undefined && !res.coordMask[aIdx]) {
      res.coordMask[aIdx] = 1;
      res.coordsAng[aIdx * 3] = x;
      res.coordsAng[aIdx * 3 + 1] = y;
      res.coordsAng[aIdx * 3 + 2] = z;
      res.availableAtoms.push(aIdx);
    }
  }
  const filtered = residues.filter((r) => r.availableAtoms.length > 0);
  if (!filtered.length) throw Error('No valid atom37 protein residues found in PDB');
  return filtered;
}

export function selectMotifAtoms(availableAtoms, atomSelectionMode = 'all_atom', resName = null) {
  if (atomSelectionMode === 'ca') return availableAtoms.includes(1) ? [1] : [];
  if (atomSelectionMode === 'bb3o') return [0, 1, 2, 4].filter((i) => availableAtoms.includes(i));
  if (atomSelectionMode === 'all_atom') return [...availableAtoms];
  if (atomSelectionMode === 'tip_atoms') {
    if (!resName) throw Error('resName is required for tip_atoms mode');
    const names = motifConstants.sidechain_tip_atoms[resName] ?? [];
    const out = [];
    for (const name of names) {
      const idx = motifConstants.atom_order[name];
      if (idx !== undefined && availableAtoms.includes(idx)) out.push(idx);
    }
    return out;
  }
  throw Error(`Unsupported atomSelectionMode: ${atomSelectionMode}`);
}

function cascadeSum16F32(arr, len) {
  const f = Math.fround;
  if (len === 0) return 0;
  let cur = arr;
  let n = len;
  while (n > 16) {
    const full = Math.floor(n / 16);
    const rem = n - full * 16;
    const next = new Float32Array(full + (rem > 0 ? 1 : 0));
    for (let ch = 0; ch < full; ch++) {
      let s = 0;
      const base = ch * 16;
      for (let j = 0; j < 16; j++) s = f(s + cur[base + j]);
      next[ch] = s;
    }
    if (rem > 0) {
      let s = 0;
      const base = full * 16;
      for (let j = 0; j < rem; j++) s = f(s + cur[base + j]);
      next[full] = s;
    }
    cur = next;
    n = next.length;
  }
  let total = 0;
  for (let j = 0; j < n; j++) total = f(total + cur[j]);
  return total;
}

function centerMaskedAtom37InPlace(coordsNm, mask) {
  const f = Math.fround;
  const totalAtoms = mask.length;
  let count = 0;
  for (let a = 0; a < totalAtoms; a++) if (mask[a]) count++;
  if (count === 0) {
    coordsNm.fill(0);
    return;
  }
  const groups = Math.floor(totalAtoms / 4);
  const laneBuf = new Float32Array(Math.max(1, groups));
  const mean = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    const lanes = new Float32Array(4);
    for (let lane = 0; lane < 4; lane++) {
      for (let g = 0; g < groups; g++) {
        const a = g * 4 + lane;
        laneBuf[g] = mask[a] ? coordsNm[a * 3 + k] : 0;
      }
      lanes[lane] = cascadeSum16F32(laneBuf, groups);
    }
    for (let a = groups * 4; a < totalAtoms; a++) {
      lanes[0] = f(lanes[0] + (mask[a] ? coordsNm[a * 3 + k] : 0));
    }
    const sum = f(f(f(lanes[0] + lanes[1]) + lanes[2]) + lanes[3]);
    mean[k] = f(sum / count);
  }
  for (let a = 0; a < totalAtoms; a++) {
    if (mask[a]) {
      coordsNm[a * 3] = f(coordsNm[a * 3] - mean[0]);
      coordsNm[a * 3 + 1] = f(coordsNm[a * 3 + 1] - mean[1]);
      coordsNm[a * 3 + 2] = f(coordsNm[a * 3 + 2] - mean[2]);
    } else {
      coordsNm[a * 3] = 0;
      coordsNm[a * 3 + 1] = 0;
      coordsNm[a * 3 + 2] = 0;
    }
  }
}

export function extractStructureForAE(input = {}, { centerCa = true } = {}) {
  const f = Math.fround;
  if (typeof input === 'string' || input.pdb) {
    const pdb = typeof input === 'string' ? input : input.pdb;
    const residues = parsePDB(pdb);
    const L = residues.length;
    const coordsNm = new Float32Array(L * 37 * 3);
    const coordMask = new Float32Array(L * 37);
    const residueTypes = new Float32Array(L);
    for (let i = 0; i < L; i++) {
      const r = residues[i];
      residueTypes[i] = r.resType;
      for (let a = 0; a < 37; a++) {
        coordMask[i * 37 + a] = r.coordMask[a];
        if (r.coordMask[a]) {
          for (let k = 0; k < 3; k++) {
            coordsNm[(i * 37 + a) * 3 + k] = f(r.coordsAng[a * 3 + k] / 10);
          }
        }
      }
    }
    if (centerCa) {
      let cx = 0, cy = 0, cz = 0;
      for (let i = 0; i < L; i++) {
        cx += coordsNm[(i * 37 + 1) * 3];
        cy += coordsNm[(i * 37 + 1) * 3 + 1];
        cz += coordsNm[(i * 37 + 1) * 3 + 2];
      }
      cx = f(cx / L); cy = f(cy / L); cz = f(cz / L);
      for (let i = 0; i < L; i++) {
        for (let a = 0; a < 37; a++) {
          if (coordMask[i * 37 + a]) {
            coordsNm[(i * 37 + a) * 3] = f(coordsNm[(i * 37 + a) * 3] - cx);
            coordsNm[(i * 37 + a) * 3 + 1] = f(coordsNm[(i * 37 + a) * 3 + 1] - cy);
            coordsNm[(i * 37 + a) * 3 + 2] = f(coordsNm[(i * 37 + a) * 3 + 2] - cz);
          }
        }
      }
    }
    const chainBreaks = new Float32Array(L);
    for (let i = 0; i < L - 1; i++) {
      const dx = (coordsNm[((i + 1) * 37 + 1) * 3] - coordsNm[(i * 37 + 1) * 3]) * 10;
      const dy = (coordsNm[((i + 1) * 37 + 1) * 3 + 1] - coordsNm[(i * 37 + 1) * 3 + 1]) * 10;
      const dz = (coordsNm[((i + 1) * 37 + 1) * 3 + 2] - coordsNm[(i * 37 + 1) * 3 + 2]) * 10;
      chainBreaks[i] = Math.hypot(dx, dy, dz) > 4.0 ? 1 : 0;
    }
    const ca = new Float32Array(L * 3);
    for (let i = 0; i < L; i++) {
      for (let k = 0; k < 3; k++) ca[i * 3 + k] = coordsNm[(i * 37 + 1) * 3 + k];
    }
    return { length: L, coordsNm, coordMask, residueTypes, chainBreaks, ca };
  }

  const L = input.length;
  if (!Number.isInteger(L) || L < 1 || L > 500) {
    throw Error('length must be an integer in [1, 500]');
  }
  const coordsNm = Float32Array.from(validateArray(input.coordsNm, L * 37 * 3, 'coordsNm'));
  const coordMask = Float32Array.from(validateArray(input.coordMask, L * 37, 'coordMask'));
  const residueTypes = Float32Array.from(validateArray(input.residueTypes, L, 'residueTypes'));
  let chainBreaks;
  if (input.chainBreaks) {
    chainBreaks = Float32Array.from(validateArray(input.chainBreaks, L, 'chainBreaks'));
  } else {
    chainBreaks = new Float32Array(L);
    for (let i = 0; i < L - 1; i++) {
      const dx = (coordsNm[((i + 1) * 37 + 1) * 3] - coordsNm[(i * 37 + 1) * 3]) * 10;
      const dy = (coordsNm[((i + 1) * 37 + 1) * 3 + 1] - coordsNm[(i * 37 + 1) * 3 + 1]) * 10;
      const dz = (coordsNm[((i + 1) * 37 + 1) * 3 + 2] - coordsNm[(i * 37 + 1) * 3 + 2]) * 10;
      chainBreaks[i] = Math.hypot(dx, dy, dz) > 4.0 ? 1 : 0;
    }
  }
  const ca = new Float32Array(L * 3);
  for (let i = 0; i < L; i++) {
    for (let k = 0; k < 3; k++) ca[i * 3 + k] = coordMask[i * 37 + 1] ? coordsNm[(i * 37 + 1) * 3 + k] : 0;
  }
  return { length: L, coordsNm, coordMask, residueTypes, chainBreaks, ca };
}

function resolveContig(contigString, { minLength, maxLength, seed = 5 } = {}) {
  const parts = contigString.split('/').map((s) => s.trim()).filter(Boolean);
  let motifLength = 0;
  const scaffoldSpecs = [];
  for (const part of parts) {
    if (ALPHABET.includes(part[0])) {
      const body = part.slice(1);
      const [s, e] = body.includes('-') ? body.split('-').map(Number) : [Number(body), Number(body)];
      if (!Number.isInteger(s) || !Number.isInteger(e) || e < s) throw Error(`Invalid motif contig segment: ${part}`);
      motifLength += e - s + 1;
    } else if (part.includes('-')) {
      const [lo, hi] = part.split('-').map(Number);
      if (!Number.isInteger(lo) || !Number.isInteger(hi) || lo < 0 || hi < lo) {
        throw Error(`Invalid scaffold range: ${part}`);
      }
      scaffoldSpecs.push([lo, hi]);
    } else {
      const n = Number(part);
      if (!Number.isInteger(n) || n < 0) throw Error(`Invalid scaffold length: ${part}`);
      scaffoldSpecs.push([n, n]);
    }
  }

  let chosenScaffold;
  if (scaffoldSpecs.every(([lo, hi]) => lo === hi)) {
    chosenScaffold = scaffoldSpecs.map(([lo]) => lo);
  } else {
    const minScaf = (minLength ?? motifLength + scaffoldSpecs.reduce((a, [lo]) => a + lo, 0)) - motifLength;
    const maxScaf = (maxLength ?? motifLength + scaffoldSpecs.reduce((a, [, hi]) => a + hi, 0)) - motifLength;
    // Sample valid combination deterministically from seed
    let state = (seed >>> 0) || 1;
    const nextU = () => {
      state = (state + 0x6d2b79f5) >>> 0;
      let x = state;
      x = Math.imul(x ^ (x >>> 15), x | 1);
      x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
      return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
    let best = null;
    for (let attempt = 0; attempt < 2048; attempt++) {
      const cand = scaffoldSpecs.map(([lo, hi]) => lo + Math.floor(nextU() * (hi - lo + 1)));
      const sum = cand.reduce((a, b) => a + b, 0);
      if (sum >= minScaf && sum <= maxScaf) {
        best = cand;
        break;
      }
    }
    if (!best) {
      // Clamp greedily into [minScaf, maxScaf]
      best = scaffoldSpecs.map(([lo]) => lo);
      let sum = best.reduce((a, b) => a + b, 0);
      for (let i = 0; i < best.length && sum < minScaf; i++) {
        const add = Math.min(scaffoldSpecs[i][1] - best[i], minScaf - sum);
        best[i] += add;
        sum += add;
      }
      if (sum < minScaf || sum > maxScaf) {
        throw Error('No valid contig combination within [minLength, maxLength]');
      }
    }
    chosenScaffold = best;
  }

  let pos = 0;
  let sIdx = 0;
  const motifIndices = [];
  const resolvedParts = [];
  for (const part of parts) {
    if (ALPHABET.includes(part[0])) {
      const body = part.slice(1);
      const [s, e] = body.includes('-') ? body.split('-').map(Number) : [Number(body), Number(body)];
      const len = e - s + 1;
      for (let k = 0; k < len; k++) motifIndices.push(pos + k);
      resolvedParts.push(`${part[0]}${pos + 1}${len > 1 ? `-${pos + len}` : ''}`);
      pos += len;
    } else {
      const len = chosenScaffold[sIdx++];
      resolvedParts.push(String(len));
      pos += len;
    }
  }
  return { length: pos, motifIndices, resolvedContig: resolvedParts.join('/') };
}

export function extractMotif(options = {}) {
  const f = Math.fround;
  // Path 1: Pre-extracted full-length tensors (e.g. oracle replay or programmatic input)
  if (options.xMotif && options.motifMask && options.seqMotif) {
    const L = options.length ?? options.seqMotif.length;
    if (!Number.isInteger(L) || L < 1 || L > 500) {
      throw Error('Motif length must be an integer in [1, 500]');
    }
    const motifMask = Float32Array.from(validateArray(options.motifMask, L * 37, 'motifMask'));
    const xMotif = Float32Array.from(validateArray(options.xMotif, L * 37 * 3, 'xMotif'));
    const seqMotif = Float32Array.from(validateArray(options.seqMotif, L, 'seqMotif'));
    if (options.centerMotif) {
      centerMaskedAtom37InPlace(xMotif, motifMask);
    }
    const seqMotifMask = new Float32Array(L);
    const activeResidueIndices = [];
    for (let i = 0; i < L; i++) {
      let any = 0;
      for (let a = 0; a < 37; a++) if (motifMask[i * 37 + a] > 0) { any = 1; break; }
      seqMotifMask[i] = any;
      if (any) activeResidueIndices.push(i);
    }
    const motifIndices = options.motifIndices ? [...options.motifIndices] : activeResidueIndices;
    const M = motifIndices.length;
    const motifMaskRaw = new Float32Array(M * 37);
    const xMotifRaw = new Float32Array(M * 37 * 3);
    const residueTypeRaw = new Float32Array(M);
    for (let m = 0; m < M; m++) {
      const r = motifIndices[m];
      residueTypeRaw[m] = seqMotif[r];
      for (let a = 0; a < 37; a++) {
        motifMaskRaw[m * 37 + a] = motifMask[r * 37 + a];
        for (let k = 0; k < 3; k++) xMotifRaw[(m * 37 + a) * 3 + k] = xMotif[(r * 37 + a) * 3 + k];
      }
    }
    return {
      length: L,
      nMotifRes: activeResidueIndices.length,
      activeResidueIndices,
      motifIndices,
      resolvedContig: options.resolvedContig ?? null,
      motifMask,
      xMotif,
      seqMotif,
      seqMotifMask,
      motifMaskRaw,
      xMotifRaw,
      residueTypeRaw,
    };
  }

  // Path 2: Extract from PDB + contig or motifAtomSpec
  const {
    pdb,
    contig,
    motifOnly = false,
    atomSelectionMode = 'all_atom',
    motifAtomSpec = null,
    minLength,
    maxLength,
    seed = 5,
    centerMotif = true,
  } = options;
  if (!pdb) throw Error('Motif extraction requires pdb text or pre-extracted xMotif/motifMask/seqMotif tensors');
  const residues = parsePDB(pdb);

  let motifMaskRaw, xMotifRaw, residueTypeRaw;
  if (motifAtomSpec) {
    const specs = [];
    const re = /([A-Za-z])(\d+):\s*\[([^\]]+)\]/g;
    let m;
    while ((m = re.exec(motifAtomSpec))) {
      specs.push([m[1], Number.parseInt(m[2], 10), m[3].split(',').map((x) => x.trim()).filter(Boolean)]);
    }
    if (!specs.length) throw Error(`Invalid motifAtomSpec: ${motifAtomSpec}`);
    const unique = [];
    const seen = new Set();
    for (const [c, r] of specs) {
      const key = `${c}:${r}`;
      if (!seen.has(key)) { seen.add(key); unique.push([c, r]); }
    }
    const nRes = unique.length;
    motifMaskRaw = new Float32Array(nRes * 37);
    xMotifRaw = new Float32Array(nRes * 37 * 3);
    residueTypeRaw = new Float32Array(nRes).fill(RESTYPE_NUM);
    for (let i = 0; i < nRes; i++) {
      const [c, r] = unique[i];
      const wanted = new Set(specs.filter(([cc, rr]) => cc === c && rr === r).flatMap(([, , names]) => names));
      const res = residues.find((item) => item.chainId === c && item.resId === r);
      if (!res) continue;
      residueTypeRaw[i] = res.resType;
      for (const name of wanted) {
        const aIdx = motifConstants.atom_order[name];
        if (aIdx !== undefined && res.coordMask[aIdx]) {
          motifMaskRaw[i * 37 + aIdx] = 1;
          for (let k = 0; k < 3; k++) xMotifRaw[(i * 37 + aIdx) * 3 + k] = f(res.coordsAng[aIdx * 3 + k] / 10);
        }
      }
    }
  } else {
    if (!contig) throw Error('contig or motifAtomSpec is required when extracting motif from PDB');
    const selectedResidues = [];
    const seenChains = new Set();
    const seenRes = new Set();
    for (const part of contig.split('/').map((s) => s.trim()).filter(Boolean)) {
      if (!ALPHABET.includes(part[0])) continue;
      const chainId = part[0];
      let matched;
      if (motifOnly) {
        if (seenChains.has(chainId)) continue;
        seenChains.add(chainId);
        matched = residues.filter((r) => r.chainId === chainId);
      } else {
        const body = part.slice(1);
        const [start, end] = body.includes('-') ? body.split('-').map(Number) : [Number(body), Number(body)];
        matched = residues.filter((r) => r.chainId === chainId && r.resId >= start && r.resId <= end);
      }
      for (const r of matched) {
        const key = `${r.chainId}:${r.resId}`;
        if (!seenRes.has(key)) {
          seenRes.add(key);
          selectedResidues.push(r);
        }
      }
    }
    const nRes = selectedResidues.length;
    if (!nRes) throw Error(`No motif residues matched contig "${contig}" in PDB`);
    motifMaskRaw = new Float32Array(nRes * 37);
    xMotifRaw = new Float32Array(nRes * 37 * 3);
    residueTypeRaw = new Float32Array(nRes).fill(RESTYPE_NUM);
    for (let i = 0; i < nRes; i++) {
      const res = selectedResidues[i];
      residueTypeRaw[i] = res.resType;
      const chosen = selectMotifAtoms(res.availableAtoms, atomSelectionMode, res.resName);
      for (const aIdx of chosen) {
        motifMaskRaw[i * 37 + aIdx] = 1;
        for (let k = 0; k < 3; k++) xMotifRaw[(i * 37 + aIdx) * 3 + k] = f(res.coordsAng[aIdx * 3 + k] / 10);
      }
    }
  }

  if (centerMotif) {
    centerMaskedAtom37InPlace(xMotifRaw, motifMaskRaw);
  }

  const nMotifRaw = residueTypeRaw.length;
  let fullLength, motifIndices, resolvedContig;
  if (contig) {
    const resolved = resolveContig(contig, { minLength, maxLength, seed });
    fullLength = resolved.length;
    motifIndices = resolved.motifIndices;
    resolvedContig = resolved.resolvedContig;
    if (motifIndices.length !== nMotifRaw) {
      throw Error(`Contig motif length (${motifIndices.length}) !== extracted PDB motif residues (${nMotifRaw})`);
    }
  } else {
    fullLength = options.length ?? nMotifRaw;
    motifIndices = Array.from({ length: nMotifRaw }, (_, i) => i);
    resolvedContig = null;
  }

  if (fullLength < 1 || fullLength > 500) {
    throw Error(`Scaffold total length ${fullLength} must be in [1, 500]`);
  }

  const motifMask = new Float32Array(fullLength * 37);
  const xMotif = new Float32Array(fullLength * 37 * 3);
  const seqMotif = new Float32Array(fullLength).fill(RESTYPE_NUM);
  for (let m = 0; m < nMotifRaw; m++) {
    const dst = motifIndices[m];
    seqMotif[dst] = residueTypeRaw[m];
    for (let a = 0; a < 37; a++) {
      motifMask[dst * 37 + a] = motifMaskRaw[m * 37 + a];
      for (let k = 0; k < 3; k++) xMotif[(dst * 37 + a) * 3 + k] = xMotifRaw[(m * 37 + a) * 3 + k];
    }
  }
  if (centerMotif) {
    centerMaskedAtom37InPlace(xMotif, motifMask);
  }

  const seqMotifMask = new Float32Array(fullLength);
  const activeResidueIndices = [];
  for (let i = 0; i < fullLength; i++) {
    let any = 0;
    for (let a = 0; a < 37; a++) if (motifMask[i * 37 + a] > 0) { any = 1; break; }
    seqMotifMask[i] = any;
    if (any) activeResidueIndices.push(i);
  }

  return {
    length: fullLength,
    nMotifRes: activeResidueIndices.length,
    activeResidueIndices,
    motifIndices,
    resolvedContig,
    motifMask,
    xMotif,
    seqMotif,
    seqMotifMask,
    motifMaskRaw,
    xMotifRaw,
    residueTypeRaw,
  };
}

// 3x3 SVD via one-sided Jacobi orthogonalization for Kabsch alignment (zero external deps).
function svd3x3(H) {
  const U = [
    [H[0][0], H[0][1], H[0][2]],
    [H[1][0], H[1][1], H[1][2]],
    [H[2][0], H[2][1], H[2][2]],
  ];
  const V = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  const pairs = [[0, 1], [0, 2], [1, 2]];
  for (let sweep = 0; sweep < 20; sweep++) {
    for (const [p, q] of pairs) {
      let app = 0, aqq = 0, apq = 0;
      for (let i = 0; i < 3; i++) {
        app += U[i][p] * U[i][p];
        aqq += U[i][q] * U[i][q];
        apq += U[i][p] * U[i][q];
      }
      if (Math.abs(apq) <= 1e-15 * Math.sqrt(app * aqq)) continue;
      const tau = (aqq - app) / (2 * apq);
      const t = tau >= 0 ? 1 / (tau + Math.hypot(1, tau)) : -1 / (-tau + Math.hypot(1, tau));
      const c = 1 / Math.hypot(1, t);
      const s = t * c;
      for (let i = 0; i < 3; i++) {
        const up = U[i][p], uq = U[i][q];
        U[i][p] = c * up - s * uq;
        U[i][q] = s * up + c * uq;
        const vp = V[i][p], vq = V[i][q];
        V[i][p] = c * vp - s * vq;
        V[i][q] = s * vp + c * vq;
      }
    }
  }
  const S = [0, 0, 0];
  for (let j = 0; j < 3; j++) {
    const n = Math.hypot(U[0][j], U[1][j], U[2][j]);
    S[j] = n;
    if (n > 1e-15) {
      for (let i = 0; i < 3; i++) U[i][j] /= n;
    } else {
      for (let i = 0; i < 3; i++) U[i][j] = i === j ? 1 : 0;
    }
  }
  // Sort singular values descending so smallest is index 2 (for proper rotation reflection fix)
  const order = [0, 1, 2].sort((a, b) => S[b] - S[a]);
  const Us = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const Vs = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let j = 0; j < 3; j++) {
    for (let i = 0; i < 3; i++) {
      Us[i][j] = U[i][order[j]];
      Vs[i][j] = V[i][order[j]];
    }
  }
  const det3 = (M) =>
    M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
    M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
    M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
  if (det3(Us) < 0) for (let i = 0; i < 3; i++) Us[i][2] = -Us[i][2];
  if (det3(Vs) < 0) for (let i = 0; i < 3; i++) Vs[i][2] = -Vs[i][2];
  return { U: Us, V: Vs };
}

export function kabschRmsd(ptsA, ptsB) {
  const n = ptsA.length / 3;
  if (n === 0) return 0;
  if (n < 3) {
    let sq = 0;
    for (let i = 0; i < ptsA.length; i++) sq += (ptsA[i] - ptsB[i]) ** 2;
    return Math.sqrt(sq / n);
  }
  const meanA = [0, 0, 0], meanB = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      meanA[k] += ptsA[i * 3 + k];
      meanB[k] += ptsB[i * 3 + k];
    }
  }
  for (let k = 0; k < 3; k++) { meanA[k] /= n; meanB[k] /= n; }
  const H = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < n; i++) {
    for (let r = 0; r < 3; r++) {
      const ar = ptsA[i * 3 + r] - meanA[r];
      for (let c = 0; c < 3; c++) {
        H[r][c] += ar * (ptsB[i * 3 + c] - meanB[c]);
      }
    }
  }
  const { U, V } = svd3x3(H);
  // R = V * diag(1, 1, sign) * U^T aligns A -> B
  const R = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      R[r][c] = V[r][0] * U[c][0] + V[r][1] * U[c][1] + V[r][2] * U[c][2];
    }
  }
  const detR =
    R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) -
    R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) +
    R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
  const s3 = detR < 0 ? -1 : 1;
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      R[r][c] = V[r][0] * U[c][0] + V[r][1] * U[c][1] + s3 * V[r][2] * U[c][2];
    }
  }
  let sq = 0;
  for (let i = 0; i < n; i++) {
    const ax = ptsA[i * 3] - meanA[0];
    const ay = ptsA[i * 3 + 1] - meanA[1];
    const az = ptsA[i * 3 + 2] - meanA[2];
    for (let r = 0; r < 3; r++) {
      const aligned = R[r][0] * ax + R[r][1] * ay + R[r][2] * az + meanB[r];
      const d = aligned - ptsB[i * 3 + r];
      sq += d * d;
    }
  }
  return Math.sqrt(sq / n);
}

export function computeMotifRMSD(genResult, motifCond, { unindexed = false } = {}) {
  const L = genResult.residueTypes.length;
  const M = motifCond.residueTypeRaw.length;
  let matchedIndices;
  if (!unindexed) {
    matchedIndices = [...motifCond.motifIndices];
  } else {
    for (const matchAa of [true, false]) {
      matchedIndices = [];
      let allMatched = true;
      for (let i = 0; i < M; i++) {
        let hasAny = false;
        for (let a = 0; a < 37; a++) if (motifCond.motifMaskRaw[i * 37 + a] > 0) { hasAny = true; break; }
        if (!hasAny) { matchedIndices.push(0); continue; }
        let bestJ = -1;
        let bestRmsd = Infinity;
        for (let j = 0; j < L; j++) {
          if (matchedIndices.includes(j)) continue;
          if (matchAa && genResult.residueTypes[j] !== Math.round(motifCond.residueTypeRaw[i])) continue;
          let count = 0, sq = 0;
          for (let a = 0; a < 37; a++) {
            if (motifCond.motifMaskRaw[i * 37 + a] > 0 && genResult.atomMask[j * 37 + a] > 0) {
              for (let k = 0; k < 3; k++) {
                const d = (motifCond.xMotifRaw[(i * 37 + a) * 3 + k] - genResult.coords[(j * 37 + a) * 3 + k]) * 10;
                sq += d * d;
              }
              count++;
            }
          }
          if (count === 0) continue;
          const rmsd = Math.sqrt(sq / count);
          if (rmsd < bestRmsd) { bestRmsd = rmsd; bestJ = j; }
        }
        if (bestJ < 0) allMatched = false;
        matchedIndices.push(bestJ);
      }
      if (allMatched) break;
    }
    if (matchedIndices.some((x) => x < 0)) {
      matchedIndices = Array.from({ length: M }, (_, i) => i);
    }
  }

  const ptsGen = [];
  const ptsRef = [];
  let activeRes = 0;
  let sameRes = 0;
  for (let m = 0; m < M; m++) {
    const j = matchedIndices[m];
    let resHasAtom = false;
    for (let a = 0; a < 37; a++) {
      if (motifCond.motifMaskRaw[m * 37 + a] > 0) {
        resHasAtom = true;
        if (genResult.atomMask[j * 37 + a] > 0) {
          for (let k = 0; k < 3; k++) {
            ptsGen.push(genResult.coords[(j * 37 + a) * 3 + k] * 10);
            ptsRef.push(motifCond.xMotifRaw[(m * 37 + a) * 3 + k] * 10);
          }
        }
      }
    }
    if (resHasAtom) {
      activeRes++;
      if (genResult.residueTypes[j] === Math.round(motifCond.residueTypeRaw[m])) sameRes++;
    }
  }

  const genArr = Float64Array.from(ptsGen);
  const refArr = Float64Array.from(ptsRef);
  let unalignedSq = 0;
  const nAtoms = genArr.length / 3;
  for (let i = 0; i < genArr.length; i++) unalignedSq += (genArr[i] - refArr[i]) ** 2;
  const unaligned = nAtoms > 0 ? Math.sqrt(unalignedSq / nAtoms) : 0;
  const aligned = kabschRmsd(genArr, refArr);

  return {
    matchedIndices,
    activeMotifAtoms: nAtoms,
    motifRmsdAlignedAngstrom: aligned,
    motifRmsdUnalignedAngstrom: unaligned,
    motifSeqRecovery: activeRes > 0 ? sameRes / activeRes : 1,
  };
}
