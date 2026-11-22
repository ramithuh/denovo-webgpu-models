"""Replay the fixed fixtures with the upstream CPU sequence tokenizer."""

import json
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
from pathlib import Path

from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.utils.residue_constants import atom_order
from esm.utils.structure.protein_chain import ProteinChain


def main():
    tokenizer = EsmSequenceTokenizer()
    fixtures = json.loads(Path(__file__).with_name("fixtures.json").read_text())
    print(f"esm {version('esm')}, transformers {version('transformers')}, "
          f"tokenizers {version('tokenizers')}")
    for fixture in fixtures:
        text = tokenizer.chain_break_token.join(c["sequence"] for c in fixture["chains"])
        actual = tokenizer.encode(text, add_special_tokens=True)
        if actual != fixture["upstreamIds"]:
            raise ValueError(f"{fixture['id']} upstream IDs differ: {actual}")
        print(f"{fixture['id']}: {len(actual)} token IDs reproduced upstream")

    trp_cage = next(f for f in fixtures if f["id"] == "1L2Y")
    reference = trp_cage["backbone"]
    pdb = Path(__file__).with_name(reference["file"]).read_text()
    atom_ids = [atom_order[name] for name in ("N", "CA", "C")]
    for omit_first_n in (False, True):
        # Match the author chain, residue and atom when constructing the missing-atom control.
        text = pdb
        if omit_first_n:
            text = "\n".join(line for line in pdb.splitlines()
                             if not (line.startswith("ATOM  ") and line[21:26] == "A   1"
                                     and line[12:16].strip() == "N"))
        chain = ProteinChain.from_pdb(StringIO(text), chain_id="A")
        masks = chain.atom37_mask[:, atom_ids]
        if (chain.sequence != trp_cage["chains"][0]["sequence"] or chain.chain_id != "A"
                or chain.residue_index.tolist() != list(range(1, 21)) or any(chain.insertion_code)):
            raise ValueError("1L2Y residue identity or numbering differs")
        expected_masks = [[not (omit_first_n and i == 0), True, True] for i in range(20)]
        if masks.tolist() != expected_masks:
            raise ValueError("1L2Y N/CA/C presence masks differ")
        xyz = chain.atom37_positions[:, atom_ids][masks].astype("<f4")
        key = "firstNitrogenOmittedSha256" if omit_first_n else "allObservedSha256"
        if sha256(xyz.tobytes()).hexdigest() != reference[key]:
            raise ValueError(f"1L2Y upstream coordinates differ for {key}")
        print(f"1L2Y: {xyz.size} observed coordinate values reproduced upstream ({key})")


if __name__ == "__main__":
    main()
