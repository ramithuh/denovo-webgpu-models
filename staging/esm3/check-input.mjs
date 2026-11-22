import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { formatMultiChainFasta } from '../../../web/src/lib/engine/residue-tokens.ts';
import { parseStructureText } from '../../../web/src/lib/protein-ops/pdb.ts';

const fixtures = JSON.parse(readFileSync(new URL('./fixtures.json', import.meta.url), 'utf8'));
const residuesById = 'LAGVSERTIDPKQNFYMHWC';

for (const { id, chains, upstreamIds } of fixtures) {
  const fasta = formatMultiChainFasta(id, chains, { colonMultimer: true });
  const [header, ...lines] = fasta.split('\n');
  assert.equal(header, `>${id}`);

  const ids = [0]; // <cls>
  for (const symbol of lines.join('')) {
    if (symbol === ':') {
      ids.push(31); // ESM3's | chain break
    } else {
      const index = residuesById.indexOf(symbol);
      assert.ok(index >= 0, `Unexpected residue ${symbol} in ${id}`);
      ids.push(index + 4);
    }
  }
  ids.push(2); // <eos>
  assert.deepEqual(ids, upstreamIds, `${id} differs from the upstream tokenizer`);
  console.log(`${id} ${chains.map((c) => c.chainId).join('+')}: ${ids.length} token IDs match`);
}

const trpCage = fixtures.find((fixture) => fixture.id === '1L2Y');
const pdb = readFileSync(new URL(trpCage.backbone.file, import.meta.url), 'utf8');
for (const omitFirstN of [false, true]) {
  // The first ATOM record is the deposited N of A1. Remove exactly that record for the control.
  assert.match(pdb.split('\n', 1)[0], /^ATOM +1 +N +ASN A +1 /);
  const parsed = parseStructureText(omitFirstN ? pdb.slice(pdb.indexOf('\n') + 1) : pdb);
  assert.equal(parsed.sequence, trpCage.chains[0].sequence);
  const coordinates = [];
  for (let i = 0; i < parsed.nResidues; i++) {
    assert.deepEqual([parsed.chainIds[i], parsed.resNums[i], parsed.insCodes[i]], ['A', i + 1, '']);
    const mask = parsed.observedBackboneMask[i] & 7;
    assert.equal(mask, omitFirstN && i === 0 ? 6 : 7, `N/CA/C mask at A${i + 1}`);
    for (let atom = 0; atom < 3; atom++) {
      if (!(mask & (1 << atom))) continue;
      const start = (i * 4 + atom) * 3;
      coordinates.push(...parsed.backbone4Coords.subarray(start, start + 3));
    }
  }
  const bytes = Buffer.alloc(coordinates.length * 4);
  coordinates.forEach((value, index) => bytes.writeFloatLE(value, index * 4));
  const expected = omitFirstN ? trpCage.backbone.firstNitrogenOmittedSha256 : trpCage.backbone.allObservedSha256;
  assert.equal(createHash('sha256').update(bytes).digest('hex'), expected);
  const label = omitFirstN ? 'A1 N omitted' : 'deposited';
  console.log(`1L2Y A (${label}): 20 residue identities, 60 atom masks, ${coordinates.length} coordinate values match`);
}
