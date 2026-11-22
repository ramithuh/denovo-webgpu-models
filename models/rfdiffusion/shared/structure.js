import chemical from "./chemical-constants.js";
export function parsePDB(text) {
  const residues = [],
    map = new Map();
  for (const line of text.split(/\r?\n/)) {
    if (line.slice(0, 6).trim() !== "ATOM") continue;
    if (![" ", "A", ""].includes(line[16] ?? "")) continue;
    const chain = line.slice(21, 22),
      number = Number(line.slice(22, 26)),
      icode = line.slice(26, 27),
      name = line.slice(17, 20),
      key = chain + number + icode;
    let r = map.get(key);
    if (!r) {
      r = {
        chain,
        number,
        icode,
        name,
        sequence: chemical.aminoAcids.includes(name)
          ? chemical.aminoAcids.indexOf(name)
          : 20,
        coords: new Float32Array(42).fill(NaN),
      };
      map.set(key, r);
      residues.push(r);
    }
    const atom = line.slice(12, 16).trim(),
      a = chemical.atoms[r.sequence]
        .slice(0, 14)
        .findIndex((x) => x?.trim() === atom);
    if (a >= 0) {
      const xyz = [30, 38, 46].map((k) => Number(line.slice(k, k + 8)));
      if (!xyz.every(Number.isFinite)) throw Error("Invalid PDB coordinates");
      r.coords.set(xyz, a * 3);
    }
  }
  if (!residues.length) throw Error("PDB contains no ATOM residues");
  return { residues, map };
}
export function parseContig(contig, { pdb, random = () => 0.5 } = {}) {
  if (typeof contig !== "string" || !contig.trim())
    throw Error("Contig is required");
  const tokens = contig
      .replace(/[\[\]]/g, "")
      .trim()
      .split(/[\/\s]+/),
    residues = [];
  let chain = 0,
    idx = 0;
  for (const token of tokens) {
    if (!token) continue;
    if (token === "0") {
      chain++;
      idx += 200;
      continue;
    }
    let m = /^([A-Za-z])(-?\d+)(?:-(-?\d+))?$/.exec(token);
    if (m) {
      const start = +m[2],
        end = m[3] === undefined ? start : +m[3];
      if (end < start) throw Error("Descending motif range");
      for (let j = start; j <= end; j++) {
        const source = pdb?.residues.find(
          (r) => r.chain === m[1] && r.number === j,
        );
        if (
          !source ||
          !Array.from(source.coords.slice(0, 9)).every(Number.isFinite)
        )
          throw Error(`Missing backbone in motif ${m[1]}${j}`);
        residues.push({ source, chain, idx: idx++ });
      }
    } else {
      m = /^(\d+)(?:-(\d+))?$/.exec(token);
      if (!m) throw Error(`Invalid contig token ${token}`);
      const lo = +m[1],
        hi = +(m[2] ?? m[1]);
      if (lo < 1 || hi < lo) throw Error("Invalid contig length");
      const length =
        lo + Math.min(hi - lo, Math.floor(random() * (hi - lo + 1)));
      for (let j = 0; j < length; j++) residues.push({ chain, idx: idx++ });
    }
  }
  return residues;
}
export function writePDB(coords, sequence, { chains } = {}) {
  const L = sequence.length,
    n = coords.length / L / 3;
  if (!Number.isInteger(n) || n < 3) throw Error("Coordinate shape mismatch");
  let serial = 1,
    lines = [];
  for (let i = 0; i < L; i++) {
    const aa = typeof sequence[i] === "number" ? sequence[i] : 20,
      name = aa < 20 ? chemical.aminoAcids[aa] : "UNK",
      chain = chains?.[i] ?? "A";
    for (let a = 0; a < Math.min(n, 4); a++) {
      const v = Array.from(coords.slice((i * n + a) * 3, (i * n + a + 1) * 3));
      if (!v.every(Number.isFinite)) continue;
      const atom = [" N  ", " CA ", " C  ", " O  "][a];
      lines.push(
        `ATOM  ${String(serial++).padStart(5)} ${atom} ${name} ${chain}${String(i + 1).padStart(4)}    ${v.map((x) => x.toFixed(3).padStart(8)).join("")}  1.00  0.00          ${atom.trim()[0].padStart(2)}`,
      );
    }
  }
  return lines.join("\n") + "\nTER\nEND\n";
}
