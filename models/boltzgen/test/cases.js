export const cases = [
  {
    name: "unconditional_l8_s3",
    mode: "unconditional",
    L: 8,
    numSteps: 3,
    seed: 42,
  },
  {
    name: "unconditional_l24_s5",
    mode: "unconditional",
    L: 24,
    numSteps: 5,
    seed: 43,
  },
  {
    name: "binder_l8_t12_s3",
    mode: "binder",
    L: 8,
    numSteps: 3,
    seed: 44,
    targetFile: "fixtures/1crn-12.pdb",
  },
];
export const configurations = [
  ["diverse", "fp32"],
  ["diverse", "fp16"],
  ["diverse", "int8"],
  ["adherence", "fp32"],
];
export const recycledMotif = {
  name: "motif_l16_s5_r1",
  mode: "motif",
  L: 16,
  numSteps: 5,
  recyclingSteps: 1,
  seed: 45,
  targetFile: "fixtures/1crn-12.pdb",
  motifIndices: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
};
export const casesFor = (variant, precision) =>
  variant === "diverse" && precision === "fp16"
    ? [...cases, recycledMotif]
    : cases;
