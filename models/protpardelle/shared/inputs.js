export function chainLayout(lengths, chainResidueGap = 200) {
  if (!lengths.length || lengths.some((n) => !Number.isInteger(n) || n < 1))
    throw new Error("Chain lengths must be positive integers");
  const n = lengths.reduce((a, b) => a + b, 0),
    seqMask = new Float32Array(n).fill(1),
    residueIndex = new Float32Array(n),
    chainIndex = new Int32Array(n);
  let offset = 0;
  lengths.forEach((length, chain) => {
    for (let i = 0; i < length; i++) {
      residueIndex[offset + i] =
        chainResidueGap === 0
          ? i + 1
          : offset + i + 1 + chain * chainResidueGap;
      chainIndex[offset + i] = chain;
    }
    offset += length;
  });
  return { seqMask, residueIndex, chainIndex };
}
export function selectPositions(spec, structure) {
  const selected = [];
  for (const token of spec
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)) {
    const match = /^([A-Za-z])(\d+)(?:-(\d+))?$/.exec(token);
    if (!match) throw new Error(`Invalid position format: ${token}`);
    const chain = structure.chainIds.indexOf(match[1]);
    if (chain < 0) throw new Error(`Unknown chain ${match[1]}`);
    const start = Number(match[2]),
      end = Number(match[3] ?? match[2]);
    for (let i = 0; i < structure.aatype.length; i++)
      if (
        structure.chainIndex[i] === chain &&
        structure.originalIndex[i] >= start &&
        structure.originalIndex[i] <= end
      )
        selected.push(i);
  }
  return selected;
}
// drawInteger receives inclusive bounds. Supply recorded NumPy integer draws
// for exact replay; a seed is not a cross-language random-number contract.
export function contigToPlacement(
  contig,
  lengthRange,
  drawInteger = (lo, hi) => lo + Math.floor(Math.random() * (hi - lo + 1)),
) {
  const segments = contig.split(";").map((part) => {
    const match = /^([A-Za-z]?)(\d+)-(\d+)$/.exec(part);
    if (!match) throw new Error(`Invalid contig segment ${part}`);
    const [, chain, lo, hi] = match,
      start = Number(lo),
      end = Number(hi);
    if (end < start) throw new Error("Reversed contig interval");
    return { chain, start, end };
  });
  const motifLength = segments
      .filter((s) => s.chain)
      .reduce((n, s) => n + s.end - s.start + 1, 0),
    scaffolds = segments.filter((s) => !s.chain);
  let minimum = lengthRange[0] - motifLength,
    maximum = lengthRange[1] - motifLength;
  scaffolds.forEach((s, i) => {
    const rest = scaffolds.slice(i + 1),
      lo = Math.max(s.start, minimum - rest.reduce((n, s) => n + s.end, 0)),
      hi = Math.min(s.end, maximum - rest.reduce((n, s) => n + s.start, 0));
    if (lo > hi) throw new Error("Contig cannot fit requested length");
    s.length = drawInteger(lo, hi);
    if (!Number.isInteger(s.length) || s.length < lo || s.length > hi)
      throw new Error("Contig replay draw outside interval");
    minimum -= s.length;
    maximum -= s.length;
  });
  const indices = [],
    placement = [],
    full = [];
  let position = 0,
    previous = 0;
  segments.forEach((s, i) => {
    if (!s.chain) {
      previous = s.length;
      position += s.length;
      if (i === segments.length - 1) {
        placement.push(String(s.length));
        full.push(String(s.length));
      }
    } else {
      for (let j = 0; j < s.end - s.start + 1; j++) indices.push(position + j);
      if (position) {
        placement.push(String(previous));
        full.push(`${previous}/${s.chain}${s.start}-${s.end}`);
      } else full.push(`${s.chain}${s.start}-${s.end}`);
      placement.push(s.chain);
      position += s.end - s.start + 1;
    }
  });
  return {
    indices,
    placement: placement.join("/") || String(position),
    full: full.join("/") || String(position),
    length: position,
  };
}
