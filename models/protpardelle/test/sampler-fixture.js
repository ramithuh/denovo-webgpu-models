import { parseStructure } from "../shared/structure.js";
export function unpack(value, binary) {
  if (
    value &&
    typeof value === "object" &&
    "offset" in value &&
    "dtype" in value
  )
    return new (value.dtype === "I32" ? Int32Array : Float32Array)(
      binary.slice(value.offset, value.offset + value.length * 4),
    );
  if (Array.isArray(value)) return value.map((v) => unpack(v, binary));
  if (value && typeof value === "object")
    return Object.fromEntries(
      Object.entries(value).map(([k, v]) => [k, unpack(v, binary)]),
    );
  return value;
}
export function samplerOptions(meta, binary, pdb) {
  const record = unpack(meta, binary),
    config = meta.config,
    aa = config.allatom_cfg,
    cc = config.conditional_cfg,
    structure = parseStructure(pdb);
  const options = {
    seqMask: new Float32Array(8).fill(1),
    residueIndex: Float32Array.from({ length: 8 }, (_, i) => i + 1),
    chainIndex: new Int32Array(8),
    tipAtomConditioning: config.tip_atom_conditioning,
    temperature: config.temperature,
    topP: config.top_p,
    disallowAAs: config.disallow_aas,
    initialCoordinates: record.initial_coordinates,
    numSteps: 4,
    sChurn: 0.8,
    stepScale: config.step_scale,
    schedule: { sMin: 0.001, sMax: 2 },
    replay: record.rng_tape,
    sidechainMode: aa.sidechain_mode,
    skipMPNNProportion: aa.skip_mpnn_proportion,
    jumpSteps: aa.jump_steps,
    uniformSteps: aa.uniform_steps,
    annealSequence: aa.anneal_seq_resampling_rate,
    legacyUpstreamSchedule: true,
    conditioning: {
      crop: cc.crop_conditional_guidance.enabled,
      cropStart: cc.crop_conditional_guidance.start,
      recurrence: cc.num_recurrence_steps,
      reconstruction: cc.reconstruction_guidance.enabled,
      reconstructionStart: cc.reconstruction_guidance.start,
      reconstructionEnd: cc.reconstruction_guidance.end,
      guidanceSchedule: cc.reconstruction_guidance.schedule,
      maxScale: cc.reconstruction_guidance.max_scale,
      lossWeight: cc.reconstruction_guidance.loss_weights.motif,
      replacement: cc.replacement_guidance.enabled,
      replacementStart: cc.replacement_guidance.start,
      replacementEnd: cc.replacement_guidance.end,
    },
  };
  if (record.inputs) {
    options.seqMask = record.inputs.seq_mask;
    options.residueIndex = record.inputs.residue_index;
    options.chainIndex = record.inputs.chain_index;
  }
  if (config.stage2_cfg.enabled)
    options.stage2 = {
      numSteps: 4,
      rewindSteps: 3,
      sChurn: 0.8,
      stepScale: config.stage2_cfg.step_scale,
      skipMPNNProportion: config.stage2_cfg.skip_mpnn_proportion,
      sidechainMode: config.stage2_cfg.sidechain_mode,
    };
  if (cc.enabled)
    options.motif = {
      structure,
      hotspots: record.hotspots ? [1] : [],
      sourceIndices: [0, 1, 2],
      targetIndices: [2, 3, 4],
      strategy: cc.crop_conditional_guidance.strategy,
      translation: [config.dx, config.dy, config.dz],
    };
  if (config.partial_diffusion.enabled)
    options.partial = {
      structure,
      rewindSteps: 3,
      repack: config.partial_diffusion.repack,
    };
  return { options, record };
}

export function miniInput(call) {
  const d = call.mini_inputs;
  if (!d) return null;
  return {
    coords: d.denoised_coords,
    noiseLevel: new Float32Array(d.seq_mask.length).fill(d.noise_level[0]),
    seqMask: d.seq_mask,
    residueIndex: d.residue_index,
    chainIndex: new Int32Array(d.seq_mask.length),
    seqSelfCond: d.seq_self_cond,
    seqCropCond: d.seq_crop_cond,
  };
}
