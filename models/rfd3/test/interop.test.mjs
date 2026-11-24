import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';
import { parsePDB, writePDB, writeMMCIF } from '../shared/structure.js';
const rows = parsePDB(`ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 20.00
ATOM      2  CA  GLY A   2       4.800   2.000   3.000  1.00 20.00
HETATM    3 ZN    ZN B 101      14.235 123.004   1.000  1.00 10.00          ZN2+
`);
test('Biopython independently identifies missing-element CA as carbon after roundtrip', () => {
  const result = spawnSync('/home/ruh/miniconda3/envs/foundry/bin/python', ['-B', '-c', `
import io,json,sys
from Bio.PDB import PDBParser,MMCIFParser
v=json.load(sys.stdin)
p=PDBParser(QUIET=True).get_structure('p',io.StringIO(v['pdb']))
c=MMCIFParser(QUIET=True).get_structure('c',io.StringIO(v['cif']))
print(json.dumps({'elements':[a.element for a in p.get_atoms()], 'cif_atoms':len(list(c.get_atoms()))}))
`], { input: JSON.stringify({ pdb: writePDB(rows), cif: writeMMCIF(rows) }), encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout), { elements: ['C', 'C', 'ZN'], cif_atoms: 3 });
});
test('ProteinDesignSpace source parser reads polymer residues and zinc from generated CIF', async () => {
  const project = '/home/ruh/ramith/personal_projects/ProteinDesignSpace/web';
  const ts = (await import(pathToFileURL(project + '/node_modules/typescript/lib/typescript.js'))).default;
  const source = await readFile(project + '/src/lib/protein-ops/pdb.ts', 'utf8');
  const js = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } }).outputText;
  const parser = await import('data:text/javascript;base64,' + Buffer.from(js).toString('base64'));
  const cif = writeMMCIF(rows);
  // Explicit API asserted below; source is compiled in memory, without changing PDS.
  const parse = parser.parseStructureText;
  assert.equal(typeof parse, 'function');
  const result = parse(cif);
  assert.equal(result.nResidues, 2);
  assert.equal(result.sequence, 'AG');
  assert.deepEqual(result.sourceResidues.map(r => r.labelSeqId), [1, 2]);
  assert.deepEqual(result.diagnostics.ligandNames, ['ZN']);
  assert.match(result.pdbText, /HETATM/);
});

test('ProteinDesignSpace preserves every generated atom, including ATP and DNA coordinates',async()=>{
  const project='/home/ruh/ramith/personal_projects/ProteinDesignSpace/web',ts=(await import(pathToFileURL(project+'/node_modules/typescript/lib/typescript.js'))).default;
  const source=await readFile(project+'/src/lib/protein-ops/pdb.ts','utf8'),js=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ES2022}}).outputText;
  const parser=await import('data:text/javascript;base64,'+Buffer.from(js).toString('base64'));
  const generated=JSON.parse(await readFile(new URL('../fixtures/gpu-generation.json',import.meta.url))).results;
  for(const row of generated){const result=parser.parseStructureText(row.mmcif),expected=parsePDB(row.pdb);assert.equal(parsePDB(result.pdbText).length,expected.length,row.case);assert.equal(result.nResidues,new Set(expected.filter(r=>r.group_pdb==='ATOM'&&r.label_atom_id==='CA').map(r=>r.auth_asym_id+':'+r.auth_seq_id)).size);if(row.case==='ligand')assert.deepEqual(result.diagnostics.ligandNames,['ATP']);if(row.case==='dna')assert.ok(parsePDB(result.pdbText).some(r=>r.label_comp_id==='DA'));}
});
