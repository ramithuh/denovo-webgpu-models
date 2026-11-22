import test from "node:test";
import fs from "node:fs/promises";
import assert from "node:assert/strict";
import { parseStructure, featurize } from "../shared/lasermpnn/structure.js";
import {
  localWeights,
  oracle,
  batchFromOracle,
  compare,
  root,
} from "./helpers.mjs";
test("PDB parsing, idealization, ligand featurization and all graph edges match upstream", async () => {
  const w = await localWeights(),
    o = await oracle(),
    batch = batchFromOracle(o),
    s = parseStructure(
      await fs.readFile(new URL("example_pdbs/4jnj-1_prot.pdb", root), "utf8"),
    ),
    got = featurize(s, w.tensors, w.manifest.config);
  assert.equal(s.residues.length, o.meta.length);
  assert.equal(s.ligands.length, o.meta.ligand_atoms);
  for (const key of [
    "sequence",
    "backbone",
    "chi",
    "ligandCoords",
    "ligandNodes",
    "ppDistances",
    "lpDistances",
    "llDistances",
  ])
    compare(got[key], batch[key], key);
  for (const key of ["pp", "lp", "ll"])
    assert.deepEqual(got[key], batch[key], key);
  compare(got.phiPsi, o.read("input.phi_psi_angles"), "phi/psi");
});
