/** atoms are residue-major N, CA, C, CB, O xyz, in Angstroms. */
export function toPDB(
  { atoms, length = atoms?.length / 15 },
  { chain = "A" } = {},
) {
  if (!Number.isInteger(length) || atoms.length !== length * 15)
    throw Error("Invalid backbone coordinate shape");
  const names = ["N", "CA", "C", "CB", "O"];
  let text = "",
    serial = 1;
  for (let r = 0; r < length; r++)
    for (let a = 0; a < 5; a++) {
      const xyz = Array.from(
        atoms.subarray(r * 15 + a * 3, r * 15 + a * 3 + 3),
      );
      if (xyz.some((x) => !Number.isFinite(x)))
        throw Error("Nonfinite PDB coordinate");
      text += `ATOM  ${String(serial++).padStart(5)} ${(" " + names[a]).padEnd(4)} ALA ${chain.slice(0, 1)}${String(r + 1).padStart(4)}    ${xyz.map((x) => x.toFixed(3).padStart(8)).join("")}  1.00  0.00          ${names[a][0].padStart(2)}\n`;
    }
  return text + "TER\nEND\n";
}
