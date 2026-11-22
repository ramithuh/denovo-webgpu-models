"""Machine-readable inventory of all published checkpoints and sampling presets."""
import ast
import hashlib
import json
import yaml
from environment import ROOT, COMMIT


def inventory():
    presets = {}
    for path in sorted((ROOT / 'src/protpardelle/configs/running').glob('*.yaml')):
        presets[path.stem] = yaml.safe_load(path.read_text())['sampling']
    examples = {}
    for path in sorted((ROOT / 'examples/sampling').glob('*.yaml')):
        data = yaml.safe_load(path.read_text())
        examples[path.stem] = {'models': data['search_space']['models'], 'partial_diffusion': data.get('partial_diffusion')}
    source = ROOT / 'src/protpardelle/core/models.py'
    tree = ast.parse(source.read_text())
    sampler = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == 'Protpardelle'
                   for n in c.body if isinstance(n, ast.FunctionDef) and n.name == 'sample')
    checkpoints = []
    for path in sorted((ROOT / '.cache/model_params/weights').glob('*.pth')):
        checkpoints.append({'name': path.stem, 'byteLength': path.stat().st_size,
                            'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()})
    return {'upstream_commit': COMMIT, 'checkpoints': checkpoints, 'presets': presets,
            'examples': examples, 'sampler_arguments': [a.arg for a in sampler.args.args if a.arg != 'self'],
            'validation_status': 'pending', 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}


if __name__ == '__main__':
    result = inventory()
    (ROOT / 'fixtures/mode_matrix.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"{len(result['checkpoints'])} checkpoints, {len(result['presets'])} presets, {len(result['examples'])} example workflows")
