import { parseSafetensors, decodeTensor, diffStats } from "../shared/loader.js";
export { configurations, cases, casesFor } from "./cases.js";
export function paths(root, variant, precision, testCase) {
  const prefix = `${root}/weights/dynamic/${variant}_${precision}_${testCase.name}`;
  return {
    inputsUrl: prefix + ".inputs.safetensors",
    outputsUrl: prefix + ".outputs.safetensors",
    numSteps: testCase.numSteps,
    recyclingSteps: testCase.recyclingSteps ?? 0,
  };
}
export async function verifyResult(result, files, read) {
  const expected = parseSafetensors(await read(files.outputsUrl)),
    inputs = parseSafetensors(await read(files.inputsUrl)),
    mask = decodeTensor(inputs["f:atom_pad_mask"]),
    stats = {};
  const trim = (data) => {
    const values = [];
    for (let i = 0; i < mask.length; i++)
      if (mask[i]) values.push(...data.subarray(i * 3, i * 3 + 3));
    return values;
  };
  for (const name of ["atom_coords", "trajectory", "denoised"]) {
    const refs = Array.isArray(result.structure[name])
      ? result.structure[name]
      : [result.structure[name]];
    refs.forEach((r, i) => {
      const key = name === "atom_coords" ? name : `${name}.${i}`,
        a = result.outputs[r.ref],
        b = decodeTensor(expected[key]);
      // The public atom_coords contract is [N_atoms,3]. Window padding is not an atom.
      // Keep its error as a diagnostic; validate every real and designed atom slot.
      stats[key] = {
        ...diffStats(trim(a), trim(b)),
        paddedRelRMS: diffStats(a, b).relRMS,
      };
      if (stats[key].relRMS >= 1e-4)
        throw Error(`Parity failed for ${key}: ${JSON.stringify(stats[key])}`);
    });
  }
  return stats;
}
export async function verifyGenerated(result, files, read) {
  const expected = parseSafetensors(await read(files.outputsUrl)),
    inputs = parseSafetensors(await read(files.inputsUrl)),
    mask = decodeTensor(inputs["f:atom_pad_mask"]),
    stats = {};
  for (const name of ["atom_coords", "trajectory", "denoised"]) {
    const tensors = name === "atom_coords" ? [result[name]] : result[name];
    const wanted =
      name === "atom_coords"
        ? 1
        : files.numSteps + (name === "trajectory" ? 2 : 0);
    if (tensors.length !== wanted)
      throw Error(`Wrong ${name} length: ${tensors.length} vs ${wanted}`);
    for (let i = 0; i < tensors.length; i++) {
      const key = name === "atom_coords" ? name : `${name}.${i}`,
        reference = decodeTensor(expected[key]),
        trim = [];
      for (let a = 0; a < mask.length; a++)
        if (mask[a]) trim.push(...reference.subarray(a * 3, a * 3 + 3));
      stats[key] = diffStats(tensors[i], trim);
      if (stats[key].relRMS >= 1e-4)
        throw Error(`Parity failed for ${key}: ${JSON.stringify(stats[key])}`);
    }
  }
  return stats;
}
