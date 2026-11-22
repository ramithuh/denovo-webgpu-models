# ThermoMPNN audit and coverage gate

Source: Kuhlman-Lab/ThermoMPNN commit `370f76ec62bd929f7425e311d8df04a0d094990f`, MIT. The existing upstream Python sources remain unchanged. The default trained checkpoint is the only published ThermoMPNN variant in this checkout.

## Inference graph

`transfer_model.py:TransferModel.forward` invokes `protein_mpnn_utils.py:tied_featurize` and its ProteinMPNN (not the alternative implementation in `model_utils.py`). Four backbone atoms N/CA/C/O, virtual CB, 25 distance pairs ×16 Gaussian RBFs (2–22 Å; sigma 1.25), 16 positional features, 48 nearest CA neighbors, 128 hidden channels. Three encoder blocks update nodes and edges, three decoder blocks update nodes. Exact GELU, LayerNorm epsilon 1e-5, neighbor message sum /30, dropout off, coordinate augmentation off.

`ProteinMPNN.forward` lines 1258–1260 **overwrites** the causal visibility matrix with ones. Sequence is fully visible including each scored position; decoding order and RNG do not influence predictions. Readout concatenates decoder 2, decoder 1, then the wild-type amino-acid embedding (384 channels). Each mutation is processed separately with sequence length ONE through LightAttention. Consequently softmax over its last dimension is exactly 1 and only feature-convolution kernel slice 4 contributes. Export omits the unused attention convolution and the other eight feature kernel slices. Strict reload into the full original model proves this reduction has zero FP32 output error. All three decoders remain. The otherwise unused ProteinMPNN sequence logits are retained for trace validation. The head is 384→384, then ReLU/linear 384→64→32→21, then scalar affine ddG calibration, then mutant minus wild-type score. The order of affine and subtraction is preserved.

## Exhaustive mode and input matrix (defined before JS/WGSL implementation)

| Supported behavior | Fixture / validation |
| --- | --- |
| Full saturation, L×20 canonical mutations, exact WT zeros | Every case; full original 20-per-residue loop additionally checked on monomer |
| Targeted point mutations, including identity; arbitrary list/order | Every case; original targeted calls and gathered matrix entries |
| Monomer PDB | monomer, medium |
| Monomer mmCIF | monomer_cif |
| Multi-chain PDB, joint geometric context | multichain |
| Multi-chain mmCIF, multi-character chain IDs | multichain_cif |
| Select chains before graph construction | chain_selection, chain_selection_cif |
| Insertion codes, negative author numbers, missing density gaps | residue_edge_cases, residue_edge_cases_cif |
| Missing backbone atoms, altLocs, MSE, ignored ligands/ions | residue_edge_cases, residue_edge_cases_cif |
| Best substitution; optional cysteine exclusion | Output postprocessing tests |
| CA neighbor-count centrality (upstream analysis convenience) | Output postprocessing tests |
| Multiple input structures | Caller iterates independent engine.predict calls; test repeated inference |

PDB semantics follow upstream: residue numbers sorted, insertion codes lexically sorted, missing integer numbers padded with X/zero coordinates, first occurrence per atom wins regardless of altLoc/occupancy, HETATM MSE becomes MET, other HETATM entries ignored. A residue mask is one only when all 12 coordinates are present. Inputs retain author chain/number/insertion-code identity separately from model indices. Unknown ATOM residues become X. Raw matrix entries for padded/unknown sites are kept with an explicit validity mask; published CSV/selection helpers omit such sites, matching SSM's gap omission. Missing-backbone canonical residues still have upstream predictions and are flagged structurally invalid.

The upstream parser only handles PDB and its featurizer truncates multi-character chain keys. mmCIF is a port input extension: atom_site author identifiers are mapped to the exact same tensors; independent fixtures pair mmCIF (Alpha/Beta chains) with single-character PDB references. The port preserves arbitrary-length chain IDs. All PDB models are read with first atom occurrence semantics matching upstream; no ensemble averaging.

## Unsupported tasks (not inference modes of this checkpoint)

Double mutations / epistasis and indels require the separate ThermoMPNN-D/I repositories. No sequence-only ddG, sidechain packing, sequence generation, diffusion, sampling, ligands, ions, DNA/RNA conditioning, explicit hydrogen/bond features, fixed/tied positions, omit/bias conditioning, or separate sidechain model. ProteinMPNN sampling utilities are not called by ThermoMPNN; training configuration switches are not independently trained inference modes. ProteinMPNN baseline evaluation is a different model. Ligands, water, ions and H atoms cannot condition this backbone-only network. Nonprotein ATOM residues map to X as in upstream.

Fixtures are deterministic 2OCJ backbone fragments from the upstream example: 12 residues (small), 76 residues (medium), two 8-residue chains, a selected chain, and a deliberately altered 11-position parser edge-case structure. They are validation inputs, not experimental quality benchmarks.
