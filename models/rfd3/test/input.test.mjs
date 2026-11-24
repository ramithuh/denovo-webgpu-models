import test from 'node:test';import assert from 'node:assert/strict';import {readFile}from'node:fs/promises';
import {buildInput,buildReferenceConformer,structureFromResult}from'../shared/input.js';
for(const name of ['monomer','cyclic','atomic-motif','ligand','dna','binder-hotspot','unindexed','hbond-exposure','multichain','fixed-backbone','sequence-redesign','partial','rna','atomic-triad'])test('general featurizer vs PyTorch: '+name,async()=>{
 const directory=['monomer','cyclic','atomic-motif'].includes(name)?'oracle':'oracle-pass2',oracle=JSON.parse(await readFile(new URL('../fixtures/'+directory+'/'+name+'.json',import.meta.url))),spec={...oracle.specification};if(spec.input)spec.input=await readFile(spec.input,'utf8');spec.cyclic=Boolean(spec.cyclic_chains);
 const input=buildInput({...spec,cleanupGuideposts:false});assert.deepEqual(Object.keys(input.features).sort(),Object.keys(oracle.features).sort());
 for(const[key,value]of Object.entries(oracle.features)){const actual=input.features[key];assert.equal(actual.length,value.data.length,key);const max=actual.reduce((m,v,i)=>Math.max(m,Math.abs(v-value.data[i])),0);assert.ok(max<2e-6,`${key}: ${max}`);}
 if(['atomic-motif','dna','ligand','unindexed','hbond-exposure','cyclic'].includes(name)){const out=Object.fromEntries(Object.entries(oracle.output).map(([k,v])=>[k,{shape:v.shape,data:Float32Array.from(v.data)}]));assert.ok(structureFromResult(input,out).mmcif.includes('_atom_site.group_PDB'));}
});
test('reference chemistry is copied and explicit unknown chemistry fails closed',()=>{const a=buildReferenceConformer('ATP'),b=buildReferenceConformer('ATP');a.atoms[0].position[0]=999;assert.notEqual(b.atoms[0].position[0],999);assert.throws(()=>buildReferenceConformer('UNSUPPORTED'),/supply atoms and bonds/);});
test('C2 high-level symmetry features match upstream',async()=>{
 const oracle=JSON.parse(await readFile(new URL('../fixtures/sampler-oracle.json',import.meta.url))).symmetry.features,features=buildInput({length:4,symmetry:{id:'C2'}}).features;
 for(const [key,value]of Object.entries(oracle)){if(key==='sym_transform')continue;assert.deepEqual(features[key],new Float32Array(value.data),key);}
});
test('unindexed guidepost cleanup reports pre-insertion RMSD and preserves selected output atoms',async()=>{
 const input=buildInput({length:3,structure:await readFile(new URL('../fixtures/conditioning.pdb',import.meta.url),'utf8'),unindex:'A1',fixedAtoms:{A1:'N,CA'}}),features=input.features,x=features.motif_pos.slice();
 for(let a=0;a<42;a++)x[a*3]=100+Math.floor(a/14)*100;
 for(let a=0;a<2;a++)for(let c=0;c<3;c++)x[(14+a)*3+c]=features.motif_pos[(42+a)*3+c]+(c===0?0.1:0);
 const out=structureFromResult(input,{X_L:{data:x},sequence_indices_I:{data:new Int32Array(4)}});assert.equal(out.motifAssignments[0].token,1);assert.ok(Math.abs(out.motifAssignments[0].insertionRmsd-0.1)<1e-6);assert.equal(new Set(out.rows.map(r=>r.auth_seq_id)).size,3);
 const held=out.rows.filter(r=>r.auth_seq_id==='2'&&['N','CA'].includes(r.label_atom_id));assert.equal(held.length,2);assert.ok(Math.abs(Number(held[0].cartn_x))<1e-7);assert.ok(Math.abs(Number(held[1].cartn_x)-1.458)<1e-6);
});

test('P1-1: unindexed catalytic triad assigns sidechain guideposts before VX cleanup even when all diffused tokens are ALA', async () => {
  const triad = await readFile(new URL('../fixtures/triad.pdb', import.meta.url), 'utf8');
  const input = buildInput({
    length: 6,
    structure: triad,
    unindex: 'A1,A2,A3',
    fixedAtoms: { A1: 'ND1,NE2', A2: 'OG', A3: 'OD1,OD2' },
  });
  const x = input.features.motif_pos.slice();
  // Place the 6 diffused residues far apart, then place tokens 1, 3, 5 within 0.05 Å of HIS(ND1,NE2), SER(OG), ASP(OD1,OD2)
  for (let t = 0; t < 6; t++) {
    for (let s = 0; s < 14; s++) {
      x[(t * 14 + s) * 3] = 200 + t * 50;
      x[(t * 14 + s) * 3 + 1] = 200;
      x[(t * 14 + s) * 3 + 2] = 200;
    }
  }
  const copyGuide = (guideToken, targetToken, slots) => {
    const gAtoms = input.atoms.map((a, i) => [a, i]).filter(([a]) => a.token === guideToken);
    assert.equal(gAtoms.length, slots.length);
    for (let k = 0; k < slots.length; k++) {
      const srcIdx = gAtoms[k][1], dstIdx = targetToken * 14 + slots[k];
      for (let c = 0; c < 3; c++) x[dstIdx * 3 + c] = input.features.motif_pos[srcIdx * 3 + c] + (c === 0 ? 0.05 : 0);
    }
  };
  // HIS: ND1=slot 6, NE2=slot 9; SER: OG=slot 5; ASP: OD1=slot 6, OD2=slot 8
  copyGuide(6, 1, [6, 9]);
  copyGuide(7, 3, [5]);
  copyGuide(8, 5, [6, 8]);
  // All diffused tokens predicted as ALA (0)
  const out = structureFromResult(input, { X_L: { data: x }, sequence_indices_I: { data: new Int32Array(9) } });
  assert.equal(out.motifAssignments.length, 3);
  assert.deepEqual(out.motifAssignments.map(a => a.token), [1, 3, 5]);
  for (const a of out.motifAssignments) assert.ok(a.insertionRmsd < 0.06, JSON.stringify(a));
  assert.equal(out.rows.find(r => r.auth_seq_id === '2').label_comp_id, 'HIS');
  assert.equal(out.rows.find(r => r.auth_seq_id === '4').label_comp_id, 'SER');
  assert.equal(out.rows.find(r => r.auth_seq_id === '6').label_comp_id, 'ASP');
  assert.ok(out.rows.some(r => r.auth_seq_id === '2' && r.label_atom_id === 'ND1'));
  assert.ok(out.rows.some(r => r.auth_seq_id === '4' && r.label_atom_id === 'OG'));
  assert.ok(out.rows.some(r => r.auth_seq_id === '6' && r.label_atom_id === 'OD2'));
});

test('P1-2: canonical ZN has finite coordinates, and non-ATP PDB/mmCIF ligands (MG, multi-atom) work with CONECT and covalent fallback', () => {
  const znRef = buildReferenceConformer('ZN');
  assert.deepEqual(znRef.atoms[0].position, [0, 0, 0]);
  const znInput = buildInput({ length: 4, components: [{ name: 'ZN', translation: [1, 2, 3] }] });
  assert.ok(znInput.origin.every(Number.isFinite));
  assert.deepEqual(znInput.origin, [1, 2, 3]);

  const mgPdb = [
    'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N',
    'ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C',
    'ATOM      3  C   ALA A   1       2.009   1.421   0.000  1.00 20.00           C',
    'ATOM      4  O   ALA A   1       1.355   2.428   0.000  1.00 20.00           O',
    'ATOM      5  CB  ALA A   1       1.988  -0.773  -1.208  1.00 20.00           C',
    'HETATM    6 MG    MG B   1       5.000   5.000   5.000  1.00 20.00          MG2+',
    'END',
  ].join('\n');
  const mgInput = buildInput({ length: 4, structure: mgPdb, ligand: 'MG' });
  assert.equal(mgInput.tokens.length, 5);
  assert.equal(mgInput.features.is_ligand[4], 1);
  assert.ok(mgInput.origin.every(Number.isFinite));

  // Multi-atom non-ATP PDB ligand WITHOUT CONECT (covalent-radius fallback) and WITH CONECT
  const ligNoConect = [
    'HETATM    1  C1  LIG L   1       0.000   0.000   0.000  1.00  0.00           C',
    'HETATM    2  C2  LIG L   1       1.520   0.000   0.000  1.00  0.00           C',
    'HETATM    3  O1  LIG L   1       2.200   1.200   0.000  1.00  0.00           O',
    'END',
  ].join('\n');
  const ligAuto = buildInput({ length: 2, structure: ligNoConect, ligand: 'LIG' });
  const nAuto = ligAuto.tokens.length;
  assert.equal(nAuto, 5);
  // C1(tok 2)-C2(tok 3) = 1.52 Å bonded, C2(tok 3)-O1(tok 4) = 1.38 Å bonded, C1-O1 = 2.51 Å not bonded
  assert.equal(ligAuto.features.token_bonds[2 * nAuto + 3], 1);
  assert.equal(ligAuto.features.token_bonds[3 * nAuto + 4], 1);
  assert.equal(ligAuto.features.token_bonds[2 * nAuto + 4], 0);

  const ligWithConect = ligNoConect.replace('END', 'CONECT    1    2\nEND');
  const ligExplicit = buildInput({ length: 2, structure: ligWithConect, ligand: 'LIG' });
  assert.equal(ligExplicit.features.token_bonds[2 * nAuto + 3], 1);
  assert.equal(ligExplicit.features.token_bonds[3 * nAuto + 4], 0);
});

test('P1-3: partialT centers exclusively on diffused atoms, pads V0..V8 from CB/CA, and sets ref_plddt=1 on diffused source residues', () => {
  const pdb = [
    'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N',
    'ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C',
    'ATOM      3  C   ALA A   1       2.009   1.421   0.000  1.00  0.00           C',
    'ATOM      4  O   ALA A   1       1.355   2.428   0.000  1.00  0.00           O',
    'ATOM      5  CB  ALA A   1       1.988  -0.773  -1.208  1.00  0.00           C',
    'ATOM      6  N   GLY A   2       4.000   0.000   0.000  1.00  0.00           N',
    'ATOM      7  CA  GLY A   2       5.458   0.000   0.000  1.00  0.00           C',
    'ATOM      8  C   GLY A   2       6.009   1.421   0.000  1.00  0.00           C',
    'ATOM      9  O   GLY A   2       5.355   2.428   0.000  1.00  0.00           O',
    'ATOM     10  N   ALA B   1      30.000   0.000   0.000  1.00  0.00           N',
    'ATOM     11  CA  ALA B   1      31.458   0.000   0.000  1.00  0.00           C',
    'ATOM     12  C   ALA B   1      32.009   1.421   0.000  1.00  0.00           C',
    'ATOM     13  O   ALA B   1      31.355   2.428   0.000  1.00  0.00           O',
    'ATOM     14  CB  ALA B   1      31.988  -0.773  -1.208  1.00  0.00           C',
    'END',
  ].join('\n');
  const input = buildInput({
    structure: pdb,
    contig: 'A1-2,/0,B1',
    partialT: 5,
    fixedAtoms: { B1: 'ALL' },
    unfixedSequence: 'A1-2',
  });
  // 1) Centered on diffused real atoms of A1-2 (mean x = (0+1.458+2.009+1.355+1.988+4+5.458+6.009+5.355)/9 = 3.0702), NOT pulled toward B1 (x=31)
  assert.ok(Math.abs(input.origin[0] - 3.070222) < 1e-4, `origin.x=${input.origin[0]}`);
  // 2) ALA V0..V8 initialized from CB (atom 4)
  for (let slot = 5; slot < 14; slot++) {
    for (let c = 0; c < 3; c++) assert.equal(input.features.motif_pos[slot * 3 + c], input.features.motif_pos[4 * 3 + c]);
  }
  // GLY CB (slot 4) and V0..V8 (slots 5..13) initialized from CA (slot 1 of token 1 -> atom 15)
  for (let slot = 4; slot < 14; slot++) {
    for (let c = 0; c < 3; c++) assert.equal(input.features.motif_pos[(14 + slot) * 3 + c], input.features.motif_pos[(14 + 1) * 3 + c]);
  }
  // 3) ref_plddt = 1 on diffused source residues A1, A2 and 0 on fixed B1
  assert.deepEqual(Array.from(input.features.ref_plddt), [1, 1, 0]);
});

test('P1-4: structureFromResult handles non-protein argmax safely via 0..19 logit masking or UNK fallback', () => {
  const input = buildInput({ length: 2 });
  const x = new Float32Array(28 * 3);
  // Without logits: nucleotide classes 23 (G) and 27 (DC) fall back to UNK without throwing
  const noLogits = structureFromResult(input, { X_L: { data: x }, sequence_indices_I: { data: new Int32Array([23, 27]) } });
  assert.equal(noLogits.rows[0].label_comp_id, 'UNK');
  assert.equal(noLogits.rows.find(r => r.auth_seq_id === '2').label_comp_id, 'UNK');
  // With logits: masks to amino acids 0..19 (token 0 -> SER=15, token 1 -> VAL=19) and writes finite B-factor entropy
  const logits = new Float32Array(64).fill(-10);
  logits[23] = 5; logits[15] = 2;
  logits[32 + 27] = 5; logits[32 + 19] = 3;
  const withLogits = structureFromResult(input, {
    X_L: { data: x },
    sequence_indices_I: { data: new Int32Array([23, 27]) },
    sequence_logits_I: { data: logits },
  });
  assert.equal(withLogits.rows[0].label_comp_id, 'SER');
  assert.equal(withLogits.rows.find(r => r.auth_seq_id === '2').label_comp_id, 'VAL');
  assert.ok(Number(withLogits.rows[0].b_iso_or_equiv) > 0);
});

test('P1-5: buildInput checks pair buffer size in bytes (*4) against maxStorageBufferBindingSize', () => {
  // L=20 -> l=280 -> l*l*16*4 = 5,017,600 bytes
  assert.throws(() => buildInput({ length: 20, maxStorageBufferBindingSize: 4_000_000 }), RangeError);
  assert.doesNotThrow(() => buildInput({ length: 20, maxStorageBufferBindingSize: 6_000_000 }));
});

test('P2-1, P2-2, P2-3, P3-2: binder hotspot origin, multi-chain target breaks, is_non_loopy, CFG referenceFeatures, and multi-char chains', () => {
  const twoChainPdb = [
    'ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N',
    'ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C',
    'ATOM      3  C   ALA A   1       2.009   1.421   0.000  1.00  0.00           C',
    'ATOM      4  O   ALA A   1       1.355   2.428   0.000  1.00  0.00           O',
    'ATOM      5  CB  ALA A   1       1.988  -0.773  -1.208  1.00  0.00           C',
    'ATOM      6  N   ALA B   1      10.000   0.000   0.000  1.00  0.00           N',
    'ATOM      7  CA  ALA B   1      11.458   0.000   0.000  1.00  0.00           C',
    'ATOM      8  C   ALA B   1      12.009   1.421   0.000  1.00  0.00           C',
    'ATOM      9  O   ALA B   1      11.355   2.428   0.000  1.00  0.00           O',
    'ATOM     10  CB  ALA B   1      11.988  -0.773  -1.208  1.00  0.00           C',
    'END',
  ].join('\n');
  // Multi-chain binder without explicit target preserves separate chains (0=binder, 1=A, 2=B) and offsets origin ~10 Å outside hotspot
  const binder = buildInput({ mode: 'binder', length: 3, structure: twoChainPdb, hotspots: { B1: 'CA,CB' } });
  assert.deepEqual(Array.from(binder.features.asym_id), [0, 0, 0, 1, 2]);
  // Hotspot COM on B1 (CA,CB) is ~(11.723, -0.3865, -0.604); origin must be ~10 Å away from hotspot COM
  const hsDist = Math.hypot(binder.origin[0] - 11.723, binder.origin[1] - (-0.3865), binder.origin[2] - (-0.604));
  assert.ok(Math.abs(hsDist - 10.0) < 0.05, `hotspot offset dist=${hsDist}`);

  // Explicit ori_token overrides origin
  const withOri = buildInput({ mode: 'binder', length: 2, structure: twoChainPdb, ori_token: [5, 6, 7] });
  assert.deepEqual(withOri.origin, [5, 6, 7]);

  // P2-2: is_non_loopy encoding (-1 when false, +1 when true, 0 on fixed motif tokens, 0 when omitted)
  const loopyFalse = buildInput({ structure: twoChainPdb, contig: '2,A1', is_non_loopy: false });
  assert.deepEqual(Array.from(loopyFalse.features.is_non_loopy), [-1, -1, 0]);
  const loopyTrue = buildInput({ structure: twoChainPdb, contig: '2,A1', is_non_loopy: true });
  assert.deepEqual(Array.from(loopyTrue.features.is_non_loopy), [1, 1, 0]);

  // P2-3: CFG referenceFeatures zeroes active_donor/acceptor/rasa and crops unindexed tokens
  const cfgIn = buildInput({
    structure: twoChainPdb, contig: '2,A1', unindex: 'B1',
    hbondDonor: { A1: 'N' }, select_buried: { A1: 'CA' }, cfgScale: 1.5,
  });
  assert.ok(cfgIn.referenceFeatures);
  assert.equal(cfgIn.features.residue_index.length, 4);
  assert.equal(cfgIn.referenceFeatures.residue_index.length, 3);
  assert.equal(cfgIn.features.active_donor.reduce((a, b) => a + b, 0), 1);
  assert.equal(cfgIn.referenceFeatures.active_donor.reduce((a, b) => a + b, 0), 0);
  assert.equal(cfgIn.referenceFeatures.ref_atomwise_rasa.reduce((a, b) => a + b, 0), 0);

  // P3-2: Multi-char chain IDs in mmCIF ('A1' res 2 vs 'A' res 12) do not collide
  const multiCharRows = [
    { group_pdb: 'ATOM', id: '1', type_symbol: 'N', label_atom_id: 'N', label_alt_id: '.', label_comp_id: 'ALA', auth_asym_id: 'A1', auth_seq_id: '2', pdbx_pdb_ins_code: '?', cartn_x: '0', cartn_y: '0', cartn_z: '0', pdbx_pdb_model_num: '1' },
    { group_pdb: 'ATOM', id: '2', type_symbol: 'C', label_atom_id: 'CA', label_alt_id: '.', label_comp_id: 'ALA', auth_asym_id: 'A1', auth_seq_id: '2', pdbx_pdb_ins_code: '?', cartn_x: '1.4', cartn_y: '0', cartn_z: '0', pdbx_pdb_model_num: '1' },
    { group_pdb: 'ATOM', id: '3', type_symbol: 'C', label_atom_id: 'C', label_alt_id: '.', label_comp_id: 'ALA', auth_asym_id: 'A1', auth_seq_id: '2', pdbx_pdb_ins_code: '?', cartn_x: '2.0', cartn_y: '1.4', cartn_z: '0', pdbx_pdb_model_num: '1' },
    { group_pdb: 'ATOM', id: '4', type_symbol: 'O', label_atom_id: 'O', label_alt_id: '.', label_comp_id: 'ALA', auth_asym_id: 'A1', auth_seq_id: '2', pdbx_pdb_ins_code: '?', cartn_x: '1.3', cartn_y: '2.4', cartn_z: '0', pdbx_pdb_model_num: '1' },
    { group_pdb: 'ATOM', id: '5', type_symbol: 'C', label_atom_id: 'CB', label_alt_id: '.', label_comp_id: 'ALA', auth_asym_id: 'A1', auth_seq_id: '2', pdbx_pdb_ins_code: '?', cartn_x: '2.0', cartn_y: '-0.8', cartn_z: '-1.2', pdbx_pdb_model_num: '1' },
    { group_pdb: 'ATOM', id: '6', type_symbol: 'N', label_atom_id: 'N', label_alt_id: '.', label_comp_id: 'ALA', auth_asym_id: 'A', auth_seq_id: '12', pdbx_pdb_ins_code: '?', cartn_x: '10', cartn_y: '0', cartn_z: '0', pdbx_pdb_model_num: '1' },
    { group_pdb: 'ATOM', id: '7', type_symbol: 'C', label_atom_id: 'CA', label_alt_id: '.', label_comp_id: 'ALA', auth_asym_id: 'A', auth_seq_id: '12', pdbx_pdb_ins_code: '?', cartn_x: '11.4', cartn_y: '0', cartn_z: '0', pdbx_pdb_model_num: '1' },
    { group_pdb: 'ATOM', id: '8', type_symbol: 'C', label_atom_id: 'C', label_alt_id: '.', label_comp_id: 'ALA', auth_asym_id: 'A', auth_seq_id: '12', pdbx_pdb_ins_code: '?', cartn_x: '12.0', cartn_y: '1.4', cartn_z: '0', pdbx_pdb_model_num: '1' },
    { group_pdb: 'ATOM', id: '9', type_symbol: 'O', label_atom_id: 'O', label_alt_id: '.', label_comp_id: 'ALA', auth_asym_id: 'A', auth_seq_id: '12', pdbx_pdb_ins_code: '?', cartn_x: '11.3', cartn_y: '2.4', cartn_z: '0', pdbx_pdb_model_num: '1' },
    { group_pdb: 'ATOM', id: '10', type_symbol: 'C', label_atom_id: 'CB', label_alt_id: '.', label_comp_id: 'ALA', auth_asym_id: 'A', auth_seq_id: '12', pdbx_pdb_ins_code: '?', cartn_x: '12.0', cartn_y: '-0.8', cartn_z: '-1.2', pdbx_pdb_model_num: '1' },
  ];
  const noCollision = buildInput({ structure: multiCharRows, contig: 'A1:2,/0,A:12' });
  assert.equal(noCollision.tokens.length, 2);
});
