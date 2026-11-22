import fs from "node:fs/promises";
import assert from "node:assert/strict";
import { loadWeights } from "../shared/lasermpnn/weights.js";
export const root = new URL("../", import.meta.url);
export const localWeights = (precision = "fp32") =>
  loadWeights("", {
    read: (n) => fs.readFile(new URL(`weights/${precision}/${n}`, root)),
  });
export async function oracle() {
  const meta = JSON.parse(
      await fs.readFile(
        new URL("fixtures/oracle_lasermpnn.json", root),
        "utf8",
      ),
    ),
    raw = await fs.readFile(new URL("fixtures/oracle_lasermpnn.bin", root));
  const read = (name) => {
    const r = meta.tensors[name];
    if (!r) throw Error(`Missing oracle tensor ${name}`);
    const buf = raw.buffer.slice(
      raw.byteOffset + r.byteOffset,
      raw.byteOffset + r.byteOffset + r.byteLength,
    );
    return r.dtype === "I32" ? new Int32Array(buf) : new Float32Array(buf);
  };
  return { meta, read };
}
export function batchFromOracle(o) {
  const r = o.read,
    idx = (k) => {
      const a = r(k),
        e = a.length / 2;
      return [Array.from(a.slice(0, e)), Array.from(a.slice(e))];
    };
  return {
    length: o.meta.length,
    backbone: r("input.backbone_coords"),
    ligandCoords: r("ligand.lig_coords"),
    ligandNodes: r("ligand.lig_nodes"),
    pp: idx("input.pr_pr_edge_index"),
    lp: idx("input.lig_pr_edge_index"),
    ll: idx("ligand.lig_lig_edge_index"),
    ppDistances: r("input.pr_pr_edge_distance"),
    lpDistances: r("input.lig_pr_edge_distance"),
    llDistances: r("ligand.lig_lig_edge_distance"),
    sequence: r("input.sequence_indices"),
    chi: r("input.chi_angles"),
    fixed: r("input.chain_mask"),
    decodingOrder: Array.from(r("input.decoding_order")),
  };
}
export function compare(got, ref, name, tolerance = 1e-4) {
  assert.equal(got.length, ref.length, `${name} length`);
  let e = 0,
    r = 0,
    maxAbs = 0;
  for (let i = 0; i < ref.length; i++) {
    if (!Number.isFinite(ref[i])) {
      assert.ok(
        Object.is(got[i], ref[i]) ||
          (Number.isNaN(got[i]) && Number.isNaN(ref[i])),
        `${name} nonfinite @ ${i}`,
      );
      continue;
    }
    assert.ok(Number.isFinite(got[i]), `${name} finite @${i}`);
    const d = got[i] - ref[i];
    e += d * d;
    r += ref[i] * ref[i];
    maxAbs = Math.max(maxAbs, Math.abs(d));
  }
  const relRMS = Math.sqrt(e / Math.max(r, 1e-30));
  console.log(name, { relRMS, maxAbs });
  assert.ok(relRMS < tolerance, `${name}: ${relRMS} >= ${tolerance}`);
  return { relRMS, maxAbs };
}
