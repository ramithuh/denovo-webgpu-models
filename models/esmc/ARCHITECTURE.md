# ESMC-300M architecture and coverage audit

Audited before implementing JS/WGSL. Upstream: Biohub/esm commit
`26b0bc2b771e3e419ea74f445a5f35cc094a1509` (MIT); weights:
`biohub/esmc-300m-2024-12`, revision `7f10b20ae75017b2dbc884070e03434515709a8d`.
The untouched sources are downloaded into ignored `.cache/esm`.

Evidence: `esm/models/esmc.py`, `esm/pretrained.py`,
`esm/layers/{transformer_stack,blocks,attention,rotary,regression_head}.py`,
`esm/tokenization/sequence_tokenizer.py`, `esm/utils/encoding.py`.

The 300M constructor uses 960 channels, 15 heads of 64, and all 30 blocks.
Each block: pre-LN → bias-free QKV projection → separate full-channel Q/K LN
→ split-half RoPE (base 10000) → bidirectional attention → output projection
→ residual / sqrt(30/36); pre-LN → 960→5120 → split SiLU multiplication
→ 2560→960 → scaled residual. LN epsilon is 1e-5. The stack's final LN
has no bias. All pre-final-LN block states are independently exposed.
MLM head: biased 960→960, exact GELU, affine LN, biased 960→64.
Embedding table is 64×960; sequence tokenizer vocabulary has 33 tokens.
No layers, heads, or parameters are pruned.

## Mode and input matrix

| Mode / option | Contract and oracle coverage |
| --- | --- |
| Embeddings | All 30 pre-final-LN hidden states, final LN, per-token and per-residue outputs, residue mean pool; both proteins |
| MLM / wild-type marginals | All 64 logits; canonical alphabet `ACDEFGHIKLMNPQRSTVWY`; L×20 log p(mutant) − log p(WT); both proteins |
| Masked marginals | Independently replace each residue with mask; collect its 64 logits and L×20 LLRs; all positions in both proteins |
| Pseudo-perplexity / naturalness | Same complete single-mask sweep; per-residue WT log probability over all 64 vocabulary logits; naturalness = mean log p; pseudo-perplexity = exp(−naturalness) |
| Mask fill | Simultaneously mask selected positions, canonical top-1 predictions and canonical-normalized probabilities; both proteins, including multiple masks |
| Attention maps | Optional all 30 head attention matrices; upstream `output_attentions=True`; both proteins |
| Hidden-layer selection | Zero-based `hiddenLayer` or all layers; never omit computation needed for downstream output |
| Token inputs | BOS=0, EOS=2, PAD=1, MASK=32; X/B/U/Z/O, gaps `.-`, chain break `|`, `_`/`<mask>`; exact upstream token fixtures |
| Padding / packed sequence IDs | Token API supports explicit per-token integer sequence IDs; attention only where IDs equal, matching upstream; pad default uses token != 1 |
| Batch / ensembles | Independent calls, or sequential batch helper; no cross-protein attention |
| Single / multiple mutants | LLR matrix supports additive multi-mutant score; no epistasis claim; validate original residue and duplicate positions |
| Structure / complexes | No coordinate or structure conditioning in ESMC; a `|` is a sequence token, not a structural complex model |
| PDB/mmCIF identifiers | Insertion codes, negative numbering, missing density, MSE/altLoc are not model inputs. PDB/mmCIF parsing intentionally not implemented: callers must explicitly supply their chosen sequence |
| Ligands / metals / DNA / RNA / H / covalent bonds | Not supported by ESMC; no molecular context inputs are accepted |
| Generation / stochastic sampling | Deterministic masked prediction; no autoregressive/diffusion sampling in upstream ESMC |
| Fixed/tied positions, omit/bias, hotspots, sidechain packing | Not upstream ESMC modes; no optional geometry model is instantiated (`n_layers_geom=0`) |

Trp-cage: `NLYIQWLKDGGPSSGRPPPS` (20 residues).
Ubiquitin: `MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG` (76).

Oracles execute unmodified upstream CPU FP32 with strict state loading.
Quantized bundles reload into that same model with dequantized FP32 weights,
separating storage quantization error from JS/WGSL arithmetic error.
Full spatial block traces are saved for unmasked and multi-mask inputs;
for every single-mask pass, every block's masked-residue vector is saved,
along with its final LN, head stages and logits. This bounds committed fixture
size while checking all layers in every member of both complete mask sweeps.
