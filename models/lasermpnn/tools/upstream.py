"""Load the unmodified checkout under its upstream Python package name."""

from pathlib import Path
import sys
import types
import torch

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("LASErMPNN")
pkg.__path__ = [str(ROOT)]
sys.modules.setdefault("LASErMPNN", pkg)
CHECKPOINT = ROOT / ".cache/lasermpnn/laser_weights_0p1A_nothing_heldout.pt"


def load(path=CHECKPOINT):
    from LASErMPNN.run_inference import load_model_from_parameter_dict

    torch.set_num_threads(2)
    return load_model_from_parameter_dict(str(path), "cpu")


def batch_for(model, params, pdb=None):
    from LASErMPNN.run_inference import ProteinComplexData, get_protein_hierview

    pdb = pdb or ROOT / "example_pdbs/4jnj-1_prot.pdb"
    data = ProteinComplexData(
        get_protein_hierview(str(pdb)),
        str(pdb.name),
        verbose=False,
        first_shell_buried_only=False,
    )
    batch = data.output_batch_data()
    batch.construct_graphs(
        model.rotamer_builder,
        model.ligand_featurizer,
        **params["model_params"]["graph_structure"],
        protein_training_noise=0.0,
        ligand_training_noise=0.0,
        build_hydrogens=True,
    )
    batch.generate_decoding_order(stack_tensors=True)
    return data, batch
