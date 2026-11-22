"""CPU FP32 execution of unmodified pinned modules with reloaded bundle weights."""
import hashlib
import json
import random
import torch
from oracle_common import ROOT, ALPHABET, A_CASES, COMMITS, model, reload_bundle, tokens

class Writer:
    def __init__(self, precision):
        self.path = ROOT / '.cache/fixtures' / (precision + '.bin')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('wb')
        self.entries = {}
    def add(self, name, tensor):
        a = tensor.detach().cpu().contiguous().numpy().astype('<f4')
        blob = a.tobytes()
        self.entries[name] = {'offset': self.file.tell(), 'byteLength': len(blob), 'shape': list(a.shape)}
        self.file.write(blob)
        return name
    def close(self):
        self.file.close()
        return {'file': '.cache/fixtures/'+self.path.name, 'byteLength': self.path.stat().st_size, 'sha256': hashlib.sha256(self.path.read_bytes()).hexdigest(), 'tensors': self.entries}

def sample(m, writer, mode):
    seq = '#' * 24 if mode == 'B' else '####ACDE####'
    x = tokens(seq)
    order = [i for i, c in enumerate(seq) if c == '#']
    rng = random.Random(812 if mode == 'B' else 923)
    rng.shuffle(order)
    uniforms = [rng.random() for _ in order]
    steps = []
    for step, (position, u) in enumerate(zip(order, uniforms)):
        logits = m(x, torch.tensor([0]))
        writer.add(f'{mode}.{step}.logits', logits[0])
        # Upstream generate_oaardm uses len(MSA_AAS)-6 = 21 categories (includes B).
        p = logits[0, position, :21].double().softmax(-1)
        choice = int(torch.searchsorted(p.cumsum(0), torch.tensor(u, dtype=torch.double)).item())
        x[0, position] = choice
        steps.append({'position': position, 'uniform': u, 'token': choice, 'logits': f'{mode}.{step}.logits'})
    return {'initialSequence': seq, 'order': order, 'uniforms': uniforms, 'steps': steps, 'tokens': x[0].tolist(), 'sequence': ''.join(ALPHABET[i] for i in x[0])}

if __name__ == '__main__':
    for precision in ['fp32', 'fp16', 'int8']:
        m, writer = model(reload_bundle(precision)), Writer(precision)
        result = {'precision': precision, 'upstream': COMMITS, 'torch': torch.__version__, 'A': []}
        with torch.inference_mode():
            for name, seq in A_CASES:
                trace, handles = {}, []
                modules = [('embedding', m.embedder.embedder), ('projection', m.embedder.up_embedder)] + [(f'block.{i}', block) for i, block in enumerate(m.embedder.layers)] + [('final_norm', m.last_norm)]
                for key, module in modules:
                    handles.append(module.register_forward_hook(lambda mod, inp, out, key=key: trace.update({key: out[0].clone()})))
                input_mask = (tokens(seq) != 30).float().unsqueeze(-1) if name == 'padding_mask' else None
                logits = m(tokens(seq), torch.tensor([0]), input_mask=input_mask)
                trace['logits'] = logits[0]
                for h in handles: h.remove()
                result['A'].append({'name': name, 'sequence': seq, 'tokens': tokens(seq)[0].tolist(),
                    'inputMask': input_mask.flatten().tolist() if input_mask is not None else None,
                    'trace': {k: writer.add(f'A.{name}.{k}', v) for k, v in trace.items()}})
            result['B'] = sample(m, writer, 'B')
            result['C'] = sample(m, writer, 'C')
        result['binary'] = writer.close()
        (ROOT / 'fixtures' / f'oracle_{precision}.json').write_text(json.dumps(result, indent=2)+'\n')
        print(precision, result['B']['sequence'], result['C']['sequence'], flush=True)
