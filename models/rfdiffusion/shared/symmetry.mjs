export function describeSymmetry(group, coordinates) {
  const match = typeof group === "string" && /^([cd])([0-9]+)$/i.exec(group);
  if (!match)
    throw new RangeError(
      "Expected a cyclic or dihedral group such as c3 or d4",
    );
  const degree = Number(match[2]);
  const dihedral = match[1].toLowerCase() === "d";
  const copies = dihedral ? 2 * degree : degree;
  if (!Number.isSafeInteger(copies) || degree < 1)
    throw new RangeError("Invalid symmetry order");
  if (
    !(coordinates instanceof Float32Array) ||
    !coordinates.length ||
    coordinates.length % 3
  ) {
    throw new TypeError("Expected a nonempty Float32Array of xyz triples");
  }
  if (!coordinates.every(Number.isFinite))
    throw new RangeError("Coordinates must be finite");
  return { degree, dihedral, copies };
}

// Per copy, store cos(theta), sin(theta) and the y/z sign for Rx(pi).
export function rotationCoefficients({ degree, dihedral, copies }) {
  const values = new Float32Array(copies * 3);
  for (let step = 0; step < degree; step++) {
    const angle = (2 * Math.PI * step) / degree;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const offset = (dihedral ? 2 * step : step) * 3;
    values.set([cos, sin, 1], offset);
    if (dihedral) values.set([cos, sin, -1], offset + 3);
  }
  return values;
}

/**
 * Replicate one fixed asymmetric unit about the origin using RFdiffusion's
 * cyclic or dihedral subunit order. Coordinates are packed xyz triples.
 * Residue and atom ordering within each copy matches the input.
 */
export function expandSymmetry(group, coordinates) {
  const shape = describeSymmetry(group, coordinates);
  const result = new Float32Array(shape.copies * coordinates.length);
  const rotations = rotationCoefficients(shape);
  for (let copy = 0; copy < shape.copies; copy++) {
    const cos = rotations[copy * 3];
    const sin = rotations[copy * 3 + 1];
    const sign = rotations[copy * 3 + 2];
    const offset = copy * coordinates.length;
    for (let i = 0; i < coordinates.length; i += 3) {
      const x = coordinates[i];
      const y = coordinates[i + 1];
      const z = coordinates[i + 2];
      const rx = cos * x - sin * y;
      const ry = sin * x + cos * y;
      result[offset + i] = rx;
      result[offset + i + 1] = sign * ry;
      result[offset + i + 2] = sign * z;
    }
  }
  return result;
}
