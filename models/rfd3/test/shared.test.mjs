import assert from "node:assert/strict";
import test from "node:test";
import { createHash } from "node:crypto";
import { readShard, parseSafetensors } from "../shared/shard.js";
import { parsePDB, writePDB, parseMMCIF, writeMMCIF } from "../shared/structure.js";

const pdb = `MODEL        1
ATOM      1  CA BALA A  -2A    -12.345   4.000   5.001  0.40 11.17           C
HETATM    2 ZN    ZN B 101      14.235 123.004   1.000  1.00 10.00          ZN2+
ENDMDL
MODEL        2
ATOM      1  CA  ALA A  -2A    -11.345   5.000   6.001  1.00  0.00           C
ENDMDL
END`;

test("PDB → mmCIF → PDB retains ATOM/HETATM identity, models and coordinates", () => {
  const rows = parsePDB(pdb);
  assert.equal(rows.length, 3);
  assert.equal(rows[0].pdbx_pdb_ins_code, "A");
  assert.equal(rows[0].label_alt_id, "B");
  assert.equal(rows[1].pdbx_formal_charge, "2");
  const result = writePDB(parseMMCIF(writeMMCIF(rows)));
  assert.equal(result.trimEnd().replaceAll(/ +\n/g, "\n"), pdb.replace("ATOM      1  CA  ALA", "ATOM      3  CA  ALA"));
  assert.throws(() => writePDB([{ ...rows[0], auth_asym_id: "long" }]), /does not fit/);
  assert.throws(() => parsePDB("ATOM      1  CA  ALA A   1"), /numeric/);
});

test("mmCIF handles unrelated loops, quoted names, whitespace and multiline metadata", () => {
  const input = `data_example
_info.description
;header
loop_
_atom_site.ignored
;
loop_
_other.id
_other.text
1 '_label'
loop_
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.label_atom_id
_atom_site.note
1.125 2 3
"O5'" 'stop_'
4 5 6 '_atom_site.tag'
;line one
line two
;
#
`;
  const rows = parseMMCIF(input);
  assert.equal(rows[0].label_atom_id, "O5'");
  assert.equal(rows[1].note, "line one\nline two");
  const again = parseMMCIF(writeMMCIF(rows));
  assert.equal(again[0].label_atom_id, rows[0].label_atom_id);
  assert.equal(again[1].note, rows[1].note);
  assert.throws(() => parseMMCIF(input.replace("4 5 6", "4 . 6")), /numeric/);
  assert.throws(() => parseMMCIF(input + "loop_\n_bad.one\n_bad.two\n1"), /Incomplete/);
});

const payload = new Uint8Array([4, 5, 6, 7]);
const meta = { byteLength: 4, sha256: createHash("sha256").update(payload).digest("hex") };
test("corrupt cached content is replaced only after a verified fetch", async () => {
  let writes = 0;
  const store = { read: () => new ArrayBuffer(4), write: bytes => {
    assert.deepEqual(new Uint8Array(bytes), payload);
    writes++;
  } };
  const options = { cache: "opfs", openStorage: () => store, fetcher: () => new Response(payload) };
  await readShard("test", meta, options);
  assert.equal(writes, 1);
  await assert.rejects(readShard("test", meta, { ...options, fetcher: () => new Response(new Uint8Array(4)) }),
    /integrity/);
  assert.equal(writes, 1);
  await assert.rejects(readShard("test", { ...meta, byteLength: 25_000_001 }, options), /length/);
});

test("in-flight cancellation prevents persistence", async () => {
  const controller = new AbortController();
  let writes = 0;
  const result = readShard("test", meta, {
    signal: controller.signal, openStorage: () => ({ read() {}, write() { writes++; } }),
    fetcher: () => {
      controller.abort();
      return new Response(payload);
    },
  });
  await assert.rejects(result, { name: "AbortError" });
  assert.equal(writes, 0);
});

test("packed safetensors views share the shard and reject malformed layouts", () => {
  const shard = header => {
    const text = new TextEncoder().encode(JSON.stringify(header));
    const bytes = new Uint8Array(8 + text.length + payload.length);
    new DataView(bytes.buffer).setBigUint64(0, BigInt(text.length), true);
    bytes.set(text, 8);
    bytes.set(payload, text.length + 8);
    return bytes.buffer;
  };
  const header = { a: { dtype: "I8", shape: [2, 2], data_offsets: [0, 4] } };
  const bytes = shard(header);
  const view = parseSafetensors(bytes).get("a");
  assert.equal(view.bytes.buffer, bytes);
  assert.deepEqual(view.bytes, payload);
  assert.throws(() => parseSafetensors(shard({ ...header, b: header.a })), /overlap/);
  assert.throws(() => parseSafetensors(shard({ a: { ...header.a, shape: [5] } })), /span/);
});


test('PDB alignment survives missing elements, CIF uses dictionary case and unique IDs', () => {
  const input = pdb.replaceAll(/           C/g, '            ');
  const rows = parsePDB(input);
  const cif = writeMMCIF(rows);
  assert.match(cif, /_atom_site.group_PDB/);
  assert.match(cif, /_atom_site.Cartn_x/);
  const round = parseMMCIF(cif);
  assert.equal(new Set(round.map(r => r.id)).size, round.length);
  assert.equal(round[0].label_seq_id, '1');
  assert.equal(writePDB(round).split('\n')[1].slice(12, 16), ' CA ');
});

test('PDB preflight rejects collisions and noninteger numbering; models are contiguous', () => {
  const rows = parsePDB(pdb);
  assert.throws(() => writePDB([rows[0], { ...rows[0], id: '9', label_asym_id: 'other' }]), /collision/);
  for (const seq of ['1.5', '0x10', '1e2', '?'])
    assert.throws(() => writePDB([{ ...rows[0], auth_seq_id: seq }]), /integer/);
  const text = writePDB([rows[0], rows[2], rows[1]]);
  assert.equal(text.match(/^MODEL/gm).length, 2);
});

test('CIF trailing tokens, quoted missing markers and numeric uncertainties', () => {
  const text = `data_x
loop_
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.note
_atom_site.other
1.23(4) -2.34(5)e2 .3
;hello
; '?'
`;
  const rows = parseMMCIF(text);
  assert.equal(rows[0].other, '?');
  assert.match(writeMMCIF(rows), /'\?'/);
  assert.equal(parseMMCIF(writeMMCIF(rows))[0].note, 'hello');
  assert.throws(() => parseMMCIF(text.replace('1.23(4)', '0x10')), /numeric/);
});

test('safetensors rejects duplicate escaped keys, metadata types and leading whitespace', () => {
  const encode = text => {
    const header = new TextEncoder().encode(text);
    const b = new Uint8Array(8 + header.length);
    new DataView(b.buffer).setBigUint64(0, BigInt(header.length), true);
    b.set(header, 8); return b.buffer;
  };
  const tensor = '{"dtype":"F32","shape":[0],"data_offsets":[0,0]}';
  assert.throws(() => parseSafetensors(encode('{"a":' + tensor + ',"a":' + tensor + '}')), /Duplicate/);
  assert.throws(() => parseSafetensors(encode(' {"__metadata__":{}}')), /start/);
  for (const v of ['[]', 'null', '{"a":1}'])
    assert.throws(() => parseSafetensors(encode('{"__metadata__":' + v + '}')), /metadata/);
});

for (const stage of ['open', 'read', 'write']) test('cancellation interrupts stalled storage ' + stage, async () => {
  const controller = new AbortController();
  const never = () => new Promise(() => {});
  const store = { read: stage === 'read' ? never : () => undefined, write: stage === 'write' ? never : () => {} };
  const pending = readShard('test', meta, { signal: controller.signal, cache: 'opfs',
    openStorage: stage === 'open' ? never : () => store, fetcher: () => new Response(payload) });
  setTimeout(() => controller.abort(), 10);
  await assert.rejects(pending, { name: 'AbortError' });
});
