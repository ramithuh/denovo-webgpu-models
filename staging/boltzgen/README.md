# BoltzGen checkpoint inventory

## Paper and PyTorch comparison

Stark et al. describe a model that predicts three-dimensional structures and generates all-atom designs. The paper also evaluates a multi-stage design and selection workflow. Those results concern predicted structures and experimental outcomes. The only local observable measured here is the checkpoint archive layout. [Paper, model and workflow](https://hannes-stark.com/assets/boltzgen.pdf)

| Upstream capability | Evidence in this preview |
| --- | --- |
| Checkpoint selection | The official CLI identifies the diverse and adherence checkpoints. The local tool verifies both pinned ZIP indexes, the archive lengths reported by the host, and the storage inventories. |
| Model execution | The PyTorch Lightning model embeds features, runs its trunk, and invokes the structure module. The directory contains a Python archive inspector. Browser weights, a WebGPU forward pass, and output-coordinate comparisons are outstanding. |
| End-to-end evaluation | There are no local structure-quality measurements or workflow-level comparisons. Only the checkpoint inventory is ready to share. |

The Python code revision used for this comparison is `6472b44e8072`. The weight repo has its own revision below. [Official checkpoint selection](https://github.com/HannesStark/boltzgen/blob/6472b44e8072187e6eaa51271ae6838245206013/src/boltzgen/cli/boltzgen.py#L136-L145) · [PyTorch forward pass](https://github.com/HannesStark/boltzgen/blob/6472b44e8072187e6eaa51271ae6838245206013/src/boltzgen/model/models/boltz.py#L506-L675)

## Inventory

The upstream BoltzGen release publishes two main design checkpoints. The table records their individual download sizes and the uncompressed storage sizes in their PyTorch ZIP indexes. This is an input to download and packaging estimates. [Upstream implementation](https://github.com/HannesStark/boltzgen) · [Pinned release](https://huggingface.co/boltzgen/boltzgen-1/tree/c1be29e1f82ffcc72264f64b993c43fb4e0d17f0)

| Checkpoint | Archive bytes | Numbered storage records | Total storage bytes | Largest storage |
| --- | ---: | ---: | ---: | ---: |
| `boltzgen1_diverse` | 1,930,847,192 | 5,405 | 1,928,704,248 | 9 MiB |
| `boltzgen1_adherence` | 1,930,858,014 | 5,405 | 1,928,704,248 | 9 MiB |

Run the following from the repo root with Python 3. The default reads two small checked-in copies of the ZIP central directories. The optional remote check retrieves the directories again with bounded HTTP range requests to the pinned revision.

```sh
python3 tools/porting/boltzgen/inspect_archive.py
python3 tools/porting/boltzgen/inspect_archive.py --remote
```

Each reference contains the original central directory and ZIP footer, starting immediately after the last local record. Together the two references occupy 115,349 bytes compressed. Both checks validate each directory's SHA-256 and inventory. The full-object SHA-256 values in [release.json](release.json) are publisher metadata. Tensor contents and end-to-end model outputs remain unverified.

A numbered storage can back multiple tensor views or contain checkpoint state beyond the model parameters. Download size, tensor count, and peak GPU memory therefore need separate measurements. [PyTorch serialization format](https://docs.pytorch.org/docs/stable/notes/serialization.html#serialized-file-format-for-torch-save)
