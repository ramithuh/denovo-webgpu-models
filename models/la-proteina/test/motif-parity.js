import { assert, compare, checkFinalOutput, checkOutput, metrics } from './parity.js';
import { extractMotif } from '../shared/motif.js';

export function ae3InputFromRef(c, ref) {
  return {
    length: c.length,
    coordsNm: ref.coords_nm,
    coordMask: ref.coord_mask,
    residueTypes: ref.residue_type,
    chainBreaks: ref.chain_breaks,
    ca: ref.ca_nm,
    eps: ref.eps,
    centerCa: false,
  };
}

export function checkAe3Output(out, ref, c, tolerance = 1e-4) {
  const result = {
    mean: compare(out.mean, ref.mean, `${c.name}:mean`, tolerance),
    logScale: compare(out.logScale, ref.log_scale, `${c.name}:logScale`, tolerance),
    z: compare(out.z, ref.z_latent, `${c.name}:z`, tolerance),
    ca: compare(out.ca, ref.ca_nm, `${c.name}:ca`, tolerance),
    coords: compare(out.coords, ref.coords, `${c.name}:coords`, tolerance),
    logits: compare(out.logits, ref.logits, `${c.name}:logits`, tolerance),
    maskedCoords: compare(out.maskedCoords, ref.masked_coords, `${c.name}:maskedCoords`, tolerance),
  };
  assert(out.residueTypes.every((x, i) => x === ref.residue_types[i]), `${c.name}: sequence argmax mismatch`);
  assert(out.atomMask.every((x, i) => x === ref.atom_mask[i]), `${c.name}: atom mask mismatch`);
  assert(out.pdb.includes('ATOM') && out.pdb.endsWith('END\n'), `${c.name}: invalid PDB output`);
  if (out.trace && ref['trace.enc.init_repr_factory']) {
    result['enc.init'] = compare(out.trace.encoder.init, ref['trace.enc.init_repr_factory'], `${c.name}:enc.init`, tolerance);
    result['enc.pair'] = compare(out.trace.encoder.pair, ref['trace.enc.pair_rep_factory'], `${c.name}:enc.pair`, tolerance);
    result['enc.layer11'] = compare(out.trace.encoder['layer.11'], ref['trace.enc.transformer_layers.11'], `${c.name}:enc.layer11`, tolerance);
    result['dec.layer11'] = compare(out.trace.decoder['layer.11'], ref['trace.dec.transformer_layers.11'], `${c.name}:dec.layer11`, tolerance);
  }
  return result;
}

export function verifyPdbMotifExtraction(pdbText, c, ref) {
  const cond = extractMotif({
    pdb: pdbText,
    contig: c.contig,
    motifOnly: c.motif_only,
    atomSelectionMode: c.motif_atom_mode,
    centerMotif: true,
  });
  assert(cond.length === c.length, `${c.name}: motif length ${cond.length} !== ${c.length}`);
  assert(cond.nMotifRes === c.n_motif_res, `${c.name}: nMotifRes ${cond.nMotifRes} !== ${c.n_motif_res}`);
  assert(cond.resolvedContig === c.resolved_contig, `${c.name}: resolvedContig ${cond.resolvedContig} !== ${c.resolved_contig}`);
  const mMask = metrics(cond.motifMask, ref.motif_mask);
  const mSeq = metrics(cond.seqMotif, ref.seq_motif);
  const mSeqMask = metrics(cond.seqMotifMask, ref.seq_motif_mask);
  const mCoors = metrics(cond.xMotif, ref.x_motif);
  assert(mMask.maxAbs === 0 && mSeq.maxAbs === 0 && mSeqMask.maxAbs === 0, `${c.name}: motif mask/seq mismatch`);
  assert(mCoors.maxAbs < 1e-6, `${c.name}: x_motif maxAbs=${mCoors.maxAbs}`);
  return cond;
}

export function motifSampleOptions(c, ref, pdbText = null) {
  const draws = Object.keys(ref)
    .filter((k) => k.startsWith('draw.'))
    .sort((a, b) => +a.slice(5) - +b.slice(5))
    .map((k) => ref[k]);
  const motif = pdbText
    ? {
        pdb: pdbText,
        contig: c.contig,
        motifOnly: c.motif_only,
        atomSelectionMode: c.motif_atom_mode,
        centerMotif: true,
      }
    : {
        length: c.length,
        xMotif: ref.x_motif,
        motifMask: ref.motif_mask,
        seqMotif: ref.seq_motif,
        motifIndices: c.motif_indices,
        resolvedContig: c.resolved_contig,
        centerMotif: false,
      };
  return {
    variant: c.model.toLowerCase(),
    length: c.length,
    steps: c.steps,
    mode: c.mode,
    initialCa: ref.initial_ca,
    initialZ: ref.initial_z,
    draws,
    motif,
    returnTrajectory: true,
  };
}

export function checkMotifFinalOutput(out, ref, c) {
  const parity = checkFinalOutput(out, ref, c);
  assert(out.motifMetrics && Number.isFinite(out.motifMetrics.motifRmsdAlignedAngstrom), `${c.name}: missing motifMetrics`);
  const rmsdDiff = Math.abs(out.motifMetrics.motifRmsdAlignedAngstrom - c.motif_rmsd_aligned_ang);
  assert(rmsdDiff < 1e-3, `${c.name}: Kabsch motif RMSD mismatch ${out.motifMetrics.motifRmsdAlignedAngstrom} vs ${c.motif_rmsd_aligned_ang}`);
  assert(Math.abs(out.motifMetrics.motifSeqRecovery - c.motif_seq_recovery) < 1e-6, `${c.name}: motifSeqRecovery mismatch`);
  return parity;
}

export function checkMotifOutput(out, ref, c) {
  const parity = checkOutput(out, ref, c);
  assert(out.motifMetrics && Number.isFinite(out.motifMetrics.motifRmsdAlignedAngstrom), `${c.name}: missing motifMetrics`);
  const rmsdDiff = Math.abs(out.motifMetrics.motifRmsdAlignedAngstrom - c.motif_rmsd_aligned_ang);
  assert(rmsdDiff < 1e-3, `${c.name}: Kabsch motif RMSD mismatch ${out.motifMetrics.motifRmsdAlignedAngstrom} vs ${c.motif_rmsd_aligned_ang}`);
  assert(Math.abs(out.motifMetrics.motifSeqRecovery - c.motif_seq_recovery) < 1e-6, `${c.name}: motifSeqRecovery mismatch`);
  return parity;
}
