export * from "./weights.js";
export * from "./sampling.js";
export function toPDB(coordinates, { chain = "A", units = "nm" } = {}) {
  if (
    !["nm", "angstrom"].includes(units) ||
    !/^[A-Za-z0-9]$/.test(chain) ||
    !coordinates?.length ||
    coordinates.length % 3
  )
    throw RangeError("Invalid PDB input");
  const lines = [
    "REMARK   Proteina v1.2 CA backbone; no sequence or side chains",
  ];
  for (let i = 0; i < coordinates.length / 3; i++) {
    if (i >= 9999) throw RangeError("PDB residue number overflow");
    const xyz = Array.from(coordinates.slice(i * 3, i * 3 + 3), (v) => {
      v *= units === "nm" ? 10 : 1;
      if (!Number.isFinite(v) || v < -999.999 || v > 9999.999)
        throw RangeError("PDB coordinate overflow");
      return v.toFixed(3).padStart(8);
    }).join("");
    lines.push(
      `ATOM  ${String(i + 1).padStart(5)}  CA  GLY ${chain}${String(i + 1).padStart(4)}    ${xyz}  1.00  0.00           C  `,
    );
  }
  return lines.join("\n") + "\nTER\nEND\n";
}
export function foldIndices(code, mapping) {
  if (!mapping)
    throw Error(
      "The checkpoint does not include a CATH vocabulary; supply the official mapping or explicit fold indices",
    );
  const parts = String(code).split(".");
  return ["C", "A", "T"].map(
    (level, i) =>
      mapping[level]?.[parts.slice(0, i + 1).join(".")] ?? [5, 43, 1472][i],
  );
}
