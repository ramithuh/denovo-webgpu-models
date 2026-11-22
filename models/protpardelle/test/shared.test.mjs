import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { parseStructure, tokenize } from "../shared/structure.js";
import { loadBundle } from "../shared/loader.js";
import { root, read } from "./helpers.mjs";
import {
  chainLayout,
  selectPositions,
  contigToPlacement,
} from "../shared/inputs.js";
test("contigs, chain layouts and fixed positions replay original preprocessing", async () => {
  const oracle = JSON.parse(
    await readFile(root + "fixtures/input_oracle.json", "utf8"),
  );
  for (const c of oracle.contigs) {
    let i = 0;
    assert.deepEqual(
      contigToPlacement(c.contig, c.range, () => c.draws[i++]),
      c.expected,
    );
    assert.equal(i, c.draws.length);
  }
  for (const c of oracle.layouts) {
    const result = chainLayout(c.lengths, c.gap);
    for (const k of Object.keys(c.expected))
      assert.deepEqual(Array.from(result[k]), c.expected[k]);
  }
  const structure = parseStructure(
    await readFile(root + "fixtures/parser_edge.pdb", "utf8"),
  );
  for (const c of oracle.selectors)
    assert.deepEqual(selectPositions(c.spec, structure), c.expected);
});
test("PDB/mmCIF features match pinned upstream including edge cases", async () => {
  const cases = JSON.parse(
    await readFile(root + "fixtures/parser_oracle.json", "utf8"),
  );
  for (const c of cases) {
    const out = parseStructure(
      await readFile(root + "fixtures/" + c.file, "utf8"),
      {
        format: c.file.endsWith(".cif") ? "cif" : "pdb",
        chainResidueGap: c.gap,
        seqIdField: "label",
      },
    );
    for (const [key, reference] of Object.entries({
      coords: "atom_positions",
      atomMask: "atom_mask",
      bFactors: "b_factors",
      aatype: "aatype",
      origAatype: "orig_aatype",
      residueIndex: "residue_index",
      originalIndex: "residue_index_orig",
      chainIndex: "chain_index",
    }))
      assert.deepEqual(
        Array.from(out[key]),
        c.features[reference].flat(Infinity),
        `${c.file}/${c.gap}/${key}`,
      );
    assert.equal(out.hetero.length, c.hetero_count);
    assert.deepEqual(
      out.hetero.map((residue) => ({
        resname: residue.resname,
        atoms: residue.atoms.map((atom) => atom.atom),
        coords: residue.atoms.map((atom) => atom.xyz.map(Math.fround)),
      })),
      c.hetero,
    );
  }
  assert.throws(() => tokenize("ABZ"), /Unsupported/);
  assert.throws(() => parseStructure(""), /no recognized/);
});
test("loader rejects corrupted shards and missing weights", async () => {
  const baseUrl = root + "weights/cc58_epoch416/int8";
  let changed = false;
  await assert.rejects(
    loadBundle({
      baseUrl,
      read: async (url) => {
        const data = await read(url);
        if (!changed && url.endsWith(".safetensors")) {
          new Uint8Array(data)[data.byteLength - 1] ^= 1;
          changed = true;
        }
        return data;
      },
    }),
    /integrity/,
  );
  await assert.rejects(
    loadBundle({ baseUrl: root + "weights/missing/fp32", read }),
    /ENOENT/,
  );
});
