import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {parseLigands,parseReceptor,parsePDB,parseMOL2,typeMolecule,parseSDF,centroid,gridCPU,prepareGrid,loadWeights} from "../shared/index.js";
import {acquire} from "../webgpu/device.js";
const pdb=await readFile("fixtures/10gs_rec.pdb","utf8"),sdf=await readFile("fixtures/10gs_lig.sdf","utf8");
const rec=parseReceptor(pdb),[pose]=parseLigands(sdf);
const fields=["group_PDB","type_symbol","auth_atom_id","auth_comp_id","auth_asym_id","auth_seq_id","pdbx_PDB_ins_code","Cartn_x","Cartn_y","Cartn_z","occupancy","pdbx_PDB_model_num"];
const cif="data_test\nloop_\n"+fields.map(s=>"_atom_site."+s).join("\n")+"\n"+parsePDB(pdb).atoms.map(a=>["ATOM",a.el,a.name,a.res,a.chain,a.seq,a.ins||"?",a.x,a.y,a.z,1,1].join(" ")).join("\n")+"\n#\n";
assert.deepEqual(parseReceptor(cif,"mmcif").atoms.map(a=>[a.channel,a.x,a.y,a.z,a.residue]),rec.atoms.map(a=>[a.channel,a.x,a.y,a.z,a.residue]));
assert.equal(parseLigands(sdf+"\n"+sdf).length,2);
const mol2=`@<TRIPOS>MOLECULE
ethanol
3 2
SMALL
NO_CHARGES
@<TRIPOS>ATOM
1 C1 0 0 0 C.3
2 C2 1.5 0 0 C.3
3 O3 2.2 1.1 0 O.3
@<TRIPOS>BOND
1 1 2 1
2 2 3 1
`;
const alcohol=typeMolecule(parseMOL2(mol2),"ligand");
assert.deepEqual(alcohol.atoms.map(a=>a.channel),[14,15,24]);
assert.equal(parseLigands(mol2+mol2,"mol2").length,2);
const v3='Ligand\n\n\n  0  0  0     0  0            999 V3000\nM  V30 BEGIN CTAB\nM  V30 BEGIN ATOM\nM  V30 5 F 0 0 0 0\nM  V30 9 C 1.3 0 0 0\nM  V30 END ATOM\nM  V30 BEGIN BOND\nM  V30 1 1 5 9\nM  V30 END BOND\nM  V30 END CTAB\nM  END\n$$$$\n';
assert.deepEqual(parseSDF(v3)[0].bonds,[[0,1,1]]);
assert.deepEqual(parseLigands(v3)[0].atoms.map(a=>a.channel),[20,15]);
assert.equal(pose.centerAtoms.length,39);
assert(centroid(pose).every(Number.isFinite));
assert.throws(()=>parseLigands("invalid"));
const one={x:0,y:0,z:0,channel:0,radius:1},far={x:50,y:0,z:0,channel:14,radius:1};
const grid=gridCPU(prepareGrid({atoms:[one]},{atoms:[far]},[11.75,11.75,11.75]));
assert.equal(grid[0],1);assert(Math.abs(grid[2]-Math.exp(-2))<1e-7);assert.equal(grid[3],0);
const data=async url=>{const b=await readFile(url);return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);};
await assert.rejects(()=>loadWeights({baseUrl:"weights/default",read:async url=>{const b=await data(url);if(url.endsWith(".safetensors")) new Uint8Array(b)[9]^=1;return b;}}),/integrity/i);
const fallbackDevice={adapterInfo:{vendor:"Google",description:"SwiftShader"}};
assert((await acquire({device:fallbackDevice})).isFallbackAdapter);
await assert.rejects(()=>acquire({device:fallbackDevice,allowSoftwareFallback:false}),/hardware/);
console.log("Structure formats, atom typing, density boundary, shard integrity and adapter policy passed.");
