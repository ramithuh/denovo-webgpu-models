"""Inspect the local OA_DM_38M archive and reproduce a fixed CPU embedding fixture."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


# Zenodo record 8045076, version 1, file oaar-38M.tar.
SIZE = 454_881_175
MD5 = "7d8959d06cc4373c220413ef00f99fab"


def audit(path):
    size = path.stat().st_size
    if size != SIZE:
        raise ValueError(f"Checkpoint size {size:,} differs from Zenodo's {SIZE:,} bytes")
    published = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as archive:
        for chunk in iter(lambda: archive.read(1024 * 1024), b""):
            published.update(chunk)
            sha256.update(chunk)
    if published.hexdigest() != MD5:
        raise ValueError("Checkpoint MD5 differs from Zenodo record 8045076")

    import torch

    # Map storages lazily. Only the seven rows used below need tensor values.
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)["model_state_dict"]
    inventory = [
        [name, str(value.dtype), list(value.shape)] for name, value in sorted(state.items())
    ]
    layout = json.dumps(inventory, separators=(",", ":")).encode("ascii")
    # IDs from upstream Tokenizer.tokenize(["MSKGEELF"]), GFP P42212 residues 1–8.
    ids = [10, 15, 8, 5, 3, 3, 9, 4]
    table = state["module.embedder.embedder.weight"]
    output = torch.nn.functional.embedding(torch.tensor(ids), table, padding_idx=28)
    output_bytes = output.detach().numpy().astype("<f4", copy=False).tobytes()
    return {
        "archive_bytes": size,
        "archive_sha256": sha256.hexdigest(),
        "model_tensors_by_dtype": dict(Counter(str(t.dtype) for t in state.values())),
        "model_elements": sum(t.numel() for t in state.values()),
        "model_bytes": sum(t.numel() * t.element_size() for t in state.values()),
        "sorted_name_dtype_shape_sha256": hashlib.sha256(layout).hexdigest(),
        "embedding_shape": list(table.shape),
        "decoder_shape": list(state["module.decoder.conv.weight"].shape),
        "input_embedding_fixture": {
            "sequence": "MSKGEELF",
            "ids": ids,
            "rows": {str(i): table[i].tolist() for i in sorted(set(ids))},
            "output_float32_le_sha256": hashlib.sha256(output_bytes).hexdigest(),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="path to the original oaar-38M.tar")
    args = parser.parse_args()
    try:
        observed = audit(args.checkpoint)
        reference = json.loads(Path(__file__).with_name("checkpoint-reference.json").read_text())
        if observed != reference:
            raise ValueError("Checkpoint inventory or input embeddings differ from checkpoint-reference.json")
        print(json.dumps(observed, indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
