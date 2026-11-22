#!/usr/bin/env python3
"""Separate PyTorch traces for identically rounded weights (runtime parity)."""

import json
import torch
from dump_oracle import ROOT, build_input
from export_ligandmpnn import load_model, reload_bundle


def main():
    torch.set_num_threads(2)
    f, _ = build_input()
    tensors = {}
    info = {}

    def put(name, value):
        tensors[name] = value.detach().cpu().clone()

    for variant in ["010", "020"]:
        for precision in ["fp16", "int8"]:
            stem = f"ligandmpnn_v_32_{variant}_25"
            model, _ = load_model(ROOT / ".cache" / f"{stem}.pt")
            model.load_state_dict(
                reload_bundle(ROOT / "weights" / stem / precision), strict=True
            )
            prefix = f"{variant}.{precision}"
            handles = []
            for name, mod in model.named_modules():
                if (
                    name.startswith(("encoder_layers.", "decoder_layers."))
                    and name.count(".") == 1
                ):

                    def hook(mod, args, out, n=name, p=prefix):
                        if isinstance(out, tuple):
                            for key, v in zip(["h_V", "h_E"], out):
                                put(f"{p}.{n}.0.{key}", v)
                        else:
                            put(f"{p}.{n}.0", out)

                    handles.append(mod.register_forward_hook(hook))
            with torch.inference_mode():
                out = model.score(f, True)
                for k in ["logits", "log_probs"]:
                    put(prefix + "." + k, out[k])
            for h in handles:
                h.remove()
    offset = 0
    with (ROOT / "fixtures/oracle_precisions.bin").open("wb") as file:
        for name, t in tensors.items():
            a = t.numpy().astype("<f4")
            raw = a.tobytes()
            info[name] = dict(
                dtype="F32", shape=list(a.shape), byteOffset=offset, byteLength=len(raw)
            )
            file.write(raw)
            offset += len(raw)
    (ROOT / "fixtures/oracle_precisions.json").write_text(
        json.dumps(dict(tensors=info), indent=2) + "\n"
    )
    print(offset, "bytes")


if __name__ == "__main__":
    main()
