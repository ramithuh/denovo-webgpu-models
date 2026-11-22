import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import {
  parseStructure,
  featurize,
  constraints,
  ALPHABET,
} from "../shared/ligandmpnn/index.js";
import { structure } from "./helpers.mjs";
const fields = [
  "group_PDB",
  "id",
  "type_symbol",
  "label_atom_id",
  "label_alt_id",
  "label_comp_id",
  "label_asym_id",
  "label_seq_id",
  "Cartn_x",
  "Cartn_y",
  "Cartn_z",
  "occupancy",
  "auth_seq_id",
  "auth_comp_id",
  "auth_asym_id",
  "auth_atom_id",
  "pdbx_PDB_ins_code",
  "pdbx_PDB_model_num",
];
function cif(pdb) {
  const lines = pdb.split("\n").filter((l) => /^(ATOM  |HETATM)/.test(l));
  return (
    "data_holo\n#\nloop_\n" +
    fields.map((x) => "_atom_site." + x).join("\n") +
    "\n" +
    lines
      .map((l, i) =>
        [
          l.slice(0, 6).trim(),
          i + 1,
          l.slice(76, 78).trim(),
          l.slice(12, 16).trim(),
          l[16].trim() || ".",
          l.slice(17, 20).trim(),
          "WRONG",
          999,
          l.slice(30, 38).trim(),
          l.slice(38, 46).trim(),
          l.slice(46, 54).trim(),
          l.slice(54, 60).trim(),
          l.slice(22, 26).trim(),
          l.slice(17, 20).trim(),
          l[21].trim(),
          l.slice(12, 16).trim(),
          l[26].trim() || "?",
          1,
        ]
          .map((v) => `'${v}'`)
          .join(" "),
      )
      .join("\n") +
    "\n#\n"
  );
}
test("mmCIF holo parser uses author chain/residue addresses and preserves DNA/zinc", () => {
  const a = parseStructure(structure),
    b = parseStructure(cif(structure));
  for (const k of [
    "X",
    "Y",
    "Y_t",
    "Y_m",
    "S",
    "R_idx",
    "chain_labels",
    "xyz_37",
    "xyz_37_m",
  ])
    assert.deepEqual(b[k], a[k], k);
  assert.deepEqual(b.residues, a.residues);
  assert.ok(b.Y_t.includes(30));
  assert.ok(b.Y_t.includes(15));
});
test("whole-chain locks, explicit redesign, omitted alphabets, and tie validation", () => {
  const p = parseStructure(structure);
  assert.ok(
    constraints(p, { fixedPositions: ["C"] }).chain_mask.every((x) => x === 0),
  );
  const c = constraints(p, {
    redesignedResidues: ["C12", "C29"],
    fixedPositions: ["C29"],
  });
  assert.equal(
    c.chain_mask.reduce((a, b) => a + b, 0),
    1,
  );
  assert.throws(
    () => constraints(p, { omitAAs: ALPHABET.slice(0, 20) }),
    /omit all/,
  );
  assert.throws(() => constraints(p, { fixedPositions: ["Z999"] }), /Unknown/);
  assert.throws(
    () =>
      constraints(p, {
        fixedPositions: ["C8"],
        tiedPositions: [["C8", "C12"]],
      }),
    /status/,
  );
  assert.throws(
    () =>
      constraints(p, {
        tiedPositions: [
          [0, 1],
          [1, 2],
        ],
      }),
    /overlapping/,
  );
});
test("MSE conversion and alternate conformer/occupancy filtering", () => {
  const original = structure
    .split("\n")
    .filter((l) => l.startsWith("ATOM") && l.slice(22, 26).trim() === "8");
  const modified = original.map((l) => l.slice(0, 17) + "MSE" + l.slice(20));
  const p = parseStructure(modified.join("\n"));
  assert.equal(p.S[0], ALPHABET.indexOf("M"));
  const b = modified.map((l) => l.slice(0, 16) + "B" + l.slice(17));
  assert.deepEqual(parseStructure([...b, ...modified].join("\n")).X, p.X);
});
test("browser modules have no Node built-in imports", () => {
  // Search only the explicitly known project source directories.
  const visit = (dir) => {
    for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
      const p = dir + "/" + ent.name;
      if (ent.isDirectory()) visit(p);
      else if (p.endsWith(".js"))
        assert.doesNotMatch(
          fs.readFileSync(p, "utf8"),
          /(?:from\s*|import\s*\()\s*['"](?:node:|fs['"/]|path['"/]|url['"/]|buffer['"/]|crypto['"/])/,
        );
    }
  };
  visit(new URL("../shared", import.meta.url).pathname);
  if (fs.existsSync(new URL("../webgpu", import.meta.url)))
    visit(new URL("../webgpu", import.meta.url).pathname);
});
