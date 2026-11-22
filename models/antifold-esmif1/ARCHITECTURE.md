# Architecture and mode audit

Reference: oxpig/AntiFold commit `9918344067f34a6b026f2634fdba212fc3e6e986`.
The reference sources under `antifold/` are kept unchanged. Both checkpoints use
the same GVP-Transformer graph. AntiFold is BSD-3-Clause; the vendored Meta ESM
implementation and GVP portions carry MIT notices (see their source headers).

## Complete inference graph

`esm/pretrained.py:IF1_dict` specifies four GVP message-passing layers with
1024 scalar / 256 vector node channels and 32 scalar / 1 vector edge channels,
30 nearest neighbors, then eight transformer encoder and eight decoder layers
(512 channels, eight heads, ReLU feedforward width 2048). All layers are retained.
GVP messages concatenate source, edge, destination and aggregate at destination
by mean. Three message GVPs and two feedforward GVPs per block; vector gates use
sigmoid; scalar gates use ReLU. Vector normalization epsilon is 1e-4 in GVP
LayerNorm, but the GVP linear transforms use their own default 1e-8.

The encoder sums masked token embeddings, normalized dihedral embeddings,
frame-rotated GVP outputs, confidence RBF embeddings, and frame-rotated raw GVP
features. It adds sinusoidal position embeddings and zeros padding rows.
Encoder and decoder are pre-LayerNorm. Decoder has causal self attention and
encoder cross attention; the final projection has 35 vocabulary entries.
DihedralFeatures uses unbiased variance and a second epsilon outside sqrt;
this is distinct from ordinary LayerNorm. Missing coordinates are zeroed only
after masks are derived. Terminal coordinates are infinity; inter-chain linkers
are ten NaN residues. Invalid-coordinate edges are removed, not merely masked.

## Mode and input matrix (verification results tracked separately)

| Mode / option | Reference behavior and intended port contract |
| --- | --- |
| AntiFold VH+VL | Heavy chain first, ten missing-coordinate gap tokens, then light chain; native sequence teacher forcing |
| AntiFold VHH | One selected heavy/nanobody chain; no artificial partner |
| Antibody/nanobody + antigen | Explicit ordered protein context chains, ten-residue linkers; preserve context residues |
| ESM-IF1 monomer | Backbone N/CA/C, confidence, native/candidate sequence |
| ESM-IF1 complex | Target chain first, other chains in file/explicit order; score/sample target with all structure context |
| Sequence likelihood | Full 35-token conditional log likelihood; all-residue and finite-coordinate means and perplexities |
| AntiFold table scores | 20 canonical amino acids, per-position entropy perplexity, region/global mean log probability, mutation scores |
| Native-conditioned CDR sampling | Independent draws from temperature-scaled 20-AA logits, matching `sample_new_sequences_CDR_HL/H`; optional expected-variation backmutation with explicit replay draws |
| Autoregressive sampling | Both checkpoints: left-to-right decoder, full vocabulary, exact recorded inverse-CDF draws; fixed tokens, temperature, encoder cached once |
| Region selection | IMGT all/FW/CDR, H/L-specific aliases, chain-prefixed ranges; insertion variants included at their integer position; exact insertion selections also accepted |
| Kabat input | Supplied Kabat numbering with explicit scheme and H/L CDR boundaries; no implicit assumption that Kabat indices are IMGT |
| Embedding extraction | Final encoder embedding and optional every GVP/transformer block trace |
| Input syntax | PDB / mmCIF model 1, signed residue IDs, insertion codes, multi-character CIF chains, first alternate conformer, MSE/SEC/UNK conversion, missing backbone atoms |
| Missing density | Missing atoms remain NaN; absent residue IDs do not create phantom residues; explicit coordinate masks supported |
| Batches/ensembles | Independent calls/samples; no training noise or dropout at inference |

Molecular ligands, ions, DNA/RNA, hydrogens, bonds, and sidechains are not model
conditioning tracks: upstream filters to peptide N/CA/C. They are ignored as
context, with parser diagnostics. No small-molecule, packing, motif-scaffolding,
hotspot, tied-position, amino-acid-bias, or unconditional architecture exists in
these checkpoints. Mutation evaluation uses the same scoring API for arbitrary
candidate sequences, including single and double mutants.

## Explicit optional sub-model boundary

`antifold/main.py` invokes `imgt_numbering.renumber_pdbs` unless
`--skip_anarcii_numbering` is set. `imgt_numbering.py` imports a separate ANARCII
model. ANARCII is not a layer of GVPTransformerModel and is not exported here.
The browser contract requires numbered structures (or a caller-supplied residue
mapping) and explicit chain roles; it does not claim automatic antibody detection
or Kabat-to-IMGT conversion. Kabat region masks are an explicit port extension,
tested separately from upstream IMGT masks. No GVP-Transformer layer is pruned.

The native-conditioned AntiFold sampling and fully autoregressive ESM sampling
are distinct APIs. Cross-language seed equality is not claimed: oracle replay
uses stored uniforms and, for backmutation, stored permutation uniforms.
