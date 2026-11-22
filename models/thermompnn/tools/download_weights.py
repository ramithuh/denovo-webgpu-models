"""Fetch only pinned official checkpoints; verify even pre-existing files."""
from pathlib import Path
import hashlib
import urllib.request
ROOT = Path(__file__).resolve().parents[1]
COMMIT = '370f76ec62bd929f7425e311d8df04a0d094990f'
FILES = {
    'models/thermoMPNN_default.pt': 'af449118ccfb4e34971d802321907ed82a8a035ae80f055bf8fc56009a4a838a',
    'vanilla_model_weights/v_48_020.pt': 'c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd',
}
def verify():
    for name, digest in FILES.items():
        path = ROOT / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            data = urllib.request.urlopen(f'https://raw.githubusercontent.com/Kuhlman-Lab/ThermoMPNN/{COMMIT}/{name}').read()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f'SHA-256 mismatch: {name}')
            path.write_bytes(data)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'SHA-256 mismatch: {name}')
        print(f'verified {name}')
if __name__ == '__main__':
    verify()
