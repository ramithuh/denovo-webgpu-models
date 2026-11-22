# EvoDiff-Seq 38M port preparation

## Paper, checkpoint and local execution

Alamdari et al. describe a ByteNet-style dilated CNN trained on 42 million UniRef50 sequences. Their order-agnostic diffusion model infers the timestep from the number of masked positions. Microsoft's `OA_DM_38M()` selects `oaar-38M.tar` and passes `diffusion_timesteps=None` to `load_sequence_checkpoint`. The resulting `ByteNetTime` skips its explicit timestep embedding. [EvoDiff paper, Figure 1 and Methods](https://openreview.net/pdf?id=qP69kXPdJM) · [Pinned OA loader](https://github.com/microsoft/evodiff/blob/33206e99446f799ec11cf6e57d66ffaec837be91/evodiff/pretrained.py#L154-L161) · [Embedding call](https://github.com/microsoft/evodiff/blob/33206e99446f799ec11cf6e57d66ffaec837be91/evodiff/model.py#L131-L140)

The workshop paper gives the small model's width as 1020. The released config specifies **1024**, which the checkpoint confirms with an input projection of shape `[1024, 8, 1]`. The pinned forward path embeds tokens into 8 channels, projects to 1024, applies 16 dilated residual blocks, then applies layer normalization and a decoder producing 31 logits per position. [Workshop paper, architecture details](https://www.mlsb.io/papers_2023/Protein_generation_with_evolutionary_diffusion_sequence_is_all_you_need.pdf) · [38M config](https://github.com/microsoft/evodiff/blob/33206e99446f799ec11cf6e57d66ffaec837be91/config/config38M.json) · [Model construction](https://github.com/microsoft/evodiff/blob/33206e99446f799ec11cf6e57d66ffaec837be91/evodiff/pretrained.py#L14-L53) · [ByteNet forward path](https://github.com/microsoft/evodiff/blob/33206e99446f799ec11cf6e57d66ffaec837be91/evodiff/model.py#L93-L173)

**Hugging Face release status.** This directory can support a reference-fixture release with checkpoint provenance and CPU parity results. Execution coverage ends at the input embedding in PyTorch and Node.js. A release advertised as a working browser model still lacks a loadable browser weight package, a forward implementation through the decoder, and a browser run comparing final logits on a fixed input. The paper's perplexity and foldability measurements belong to the upstream models. [Local checkpoint evidence](checkpoint-reference.json) · [Hugging Face model cards](https://huggingface.co/docs/hub/model-cards)

## Tokenizer and input embedding parity

From the repository root, run the offline check with Node 22.

```sh
node tools/porting/evodiff/tokenizer.test.mjs
```

`tokenizer.mjs` takes a single case-sensitive string and returns one signed 32-bit token ID per character. Its Python equivalent takes a one-element list with the string as its first entry. Neither implementation inserts boundaries. The five reference rows cover all 31 IDs, including gap (26), stop (27), mask (28), start (29), and padding (30). The Python tokenizer raised `KeyError` for each of the three invalid inputs in the fixture. The JavaScript implementation raises `RangeError` for those inputs. [Upstream tokenizer](https://github.com/microsoft/evodiff/blob/33206e99446f799ec11cf6e57d66ffaec837be91/evodiff/utils.py#L148-L171) · [Token lookup](https://github.com/microsoft/evodiff/blob/33206e99446f799ec11cf6e57d66ffaec837be91/evodiff/utils.py#L251-L255) · [Alphabet definitions](https://github.com/microsoft/protein-sequence-models/blob/af695772c4a1c056d930c95ec7e6428aa042f5cd/sequence_models/constants.py)

The reference rows were captured by executing Microsoft's pinned Python tokenizer class with the pinned alphabet definitions. To repeat that comparison in a Python environment with those revisions installed, run this from the repository root.

```python
import json
from pathlib import Path
from evodiff.utils import Tokenizer

fixture = json.loads(Path("tools/porting/evodiff/tokenizer-fixtures.json").read_text())
tokenizer = Tokenizer()
for row, ids in fixture["rows"]:
    assert tokenizer.tokenize([row]).tolist() == ids
for row in fixture["unknown"]:
    try:
        tokenizer.tokenize([row])
    except KeyError:
        continue
    raise AssertionError(f"Upstream accepted {row!r}")
```

## Checkpoint audit

Zenodo record 8045076 publishes `oaar-38M.tar` at 454,881,175 bytes with MD5 `7d8959d06cc4373c220413ef00f99fab` under CC BY 4.0. The archive includes model and optimizer state. [Alamdari et al., version 1 checkpoint record](https://zenodo.org/records/8045076) · [File metadata](https://zenodo.org/api/records/8045076/files/oaar-38M.tar)

To reproduce the model tensor inventory offline with a local copy and PyTorch 2.9, run the following from the repository root.

```sh
python tools/porting/evodiff/audit_checkpoint.py /path/to/oaar-38M.tar
```

The full archive matched the published size and MD5. The model state contained **199 float32 tensors with 37,890,327 elements**, totaling 151,561,308 bytes. The input embedding tensor has shape `[31, 8]` and the decoder weight has shape `[31, 1024, 1]`. The audit compares the calculated archive SHA-256, tensor layout, and embedding fixture with `checkpoint-reference.json` and exits nonzero on a mismatch. The layout digest covers sorted triples of original state-dictionary key, PyTorch dtype, and shape.

The numerical fixture takes residues 1–8 of green fluorescent protein, `MSKGEELF`, and reads the corresponding seven rows of `module.embedder.embedder.weight`. The Python audit runs a CPU float32 embedding lookup on the original weights. The offline JavaScript test expands the selected rows into a table, tokenizes the sequence, and compares all 256 output bytes to the CPU result via SHA-256. The fixture includes 56 weight values from the authors' CC BY 4.0 archive. Full-model activations, GPU parity, browser memory use, and throughput remain unmeasured. [GFP sequence, UniProt P42212](https://rest.uniprot.org/uniprotkb/P42212.fasta) · [EvoDiff input embedding](https://github.com/microsoft/evodiff/blob/33206e99446f799ec11cf6e57d66ffaec837be91/evodiff/model.py) · [Checkpoint attribution](https://zenodo.org/records/8045076)
