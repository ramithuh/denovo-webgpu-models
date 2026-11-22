import { makeGeometry } from "../shared/lasermpnn/geometry.js";
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import {
  parseStructure,
  featurize,
  writePdb,
} from "../shared/lasermpnn/structure.js";
import { createLaserMpnnCpuEngine } from "../cpu/lasermpnn/index.js";
import { root, localWeights, compare } from "./helpers.mjs";
const pdb = await fs.readFile(
  new URL("example_pdbs/4jnj-1_prot.pdb", root),
  "utf8",
);
const structure = parseStructure(pdb);
const all = [
  ...structure.residues.flatMap((r) => Array.from(r.atoms.values())),
  ...structure.ligands,
];
const columns = [
  "group_PDB",
  "id",
  "type_symbol",
  "label_atom_id",
  "label_comp_id",
  "label_asym_id",
  "label_seq_id",
  "Cartn_x",
  "Cartn_y",
  "Cartn_z",
  "auth_asym_id",
  "auth_seq_id",
  "auth_atom_id",
  "auth_comp_id",
  "pdbx_PDB_ins_code",
  "label_alt_id",
  "pdbx_PDB_model_num",
];
const cif =
  "data_holo\n_entry.id holo\nloop_\n" +
  columns.map((c) => "_atom_site." + c).join("\n") +
  "\n" +
  all
    .map((a, i) =>
      [
        a.record,
        i + 1,
        a.element,
        `'${a.name}'`,
        a.resname,
        a.chain,
        a.resnum,
        ...a.xyz,
        a.chain,
        a.resnum,
        `"${a.name}"`,
        a.resname,
        a.icode || "?",
        a.altloc || ".",
        1,
      ].join(" "),
    )
    .join("\n") +
  "\n#\n";
test("mmCIF auth identifiers and quoted atom fields give the same holo features as PDB", async () => {
  assert.equal(
    parseStructure("# comment\n# second comment\n" + cif).residues.length,
    115,
  );
  const w = await localWeights(),
    a = featurize(structure, w.tensors, w.manifest.config),
    b = featurize(parseStructure(cif), w.tensors, w.manifest.config);
  for (const key of [
    "backbone",
    "sequence",
    "chi",
    "ligandCoords",
    "ligandNodes",
    "ppDistances",
    "lpDistances",
    "llDistances",
  ])
    compare(a[key], b[key], "mmCIF " + key, 1e-7);
  for (const key of ["pp", "lp", "ll"]) assert.deepEqual(a[key], b[key]);
  const long = featurize(
    parseStructure(
      cif
        .split("\n")
        .map((line) => {
          if (!/^(ATOM|HETATM) /.test(line)) return line;
          const fields = line.split(/\s+/);
          fields[5] = fields[10] = "LONG_CHAIN";
          return fields.join(" ");
        })
        .join("\n"),
    ),
    w.tensors,
    w.manifest.config,
  );
  assert.equal(long.structure.residues[0].chain, "LONG_CHAIN");
  const pdb = writePdb(
    long,
    { sequence: long.sequence },
    makeGeometry(w.tensors).build(long.backbone, long.sequence, long.chi),
  );
  assert.match(pdb, /LONG_CHAIN/);
  assert.equal(parseStructure(pdb).residues.length, 115);
});
test("design controls, deterministic seed, scores, PDB output, cancellation and teardown", async () => {
  const weights = await localWeights(),
    engine = await createLaserMpnnCpuEngine({ weights });
  const subset = pdb
    .split("\n")
    .filter(
      (l) =>
        l.startsWith("HETATM") ||
        (l.startsWith("ATOM") && Number(l.slice(22, 26)) < 16),
    )
    .join("\n");
  const out = await engine.design(subset, {
    temperature: 0,
    chiTemperature: 0,
    seed: 7,
    numSequences: 2,
    fixedPositions: ["A:12"],
    omitAAs: "RNDCQEGHILKMFPSTWYV",
  });
  assert.deepEqual(out.sequences, ["GAAA", "GAAA"]);
  assert.equal(out.pdbTexts.length, 2);
  assert.equal(parseStructure(out.pdbTexts[0]).ligands.length, 31);
  assert.equal(out.chiAngles[0].length, 16);
  assert.ok(Number.isFinite(out.scores[0].totalLogLikelihood));
  const excluded = await engine.design(subset, {
    designChains: ["B"],
    seed: 5,
  });
  assert.equal(excluded.sequences[0], "GAEA");
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(engine.design(subset, { signal: controller.signal }));
  await assert.rejects(
    engine.design(subset, { omitAAs: "ARNDCQEGHILKMFPSTWYV" }),
    /every amino acid/,
  );
  engine.destroy();
  engine.destroy();
  await assert.rejects(engine.design(subset), /destroyed/);
});
