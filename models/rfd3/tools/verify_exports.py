#!/usr/bin/env python3
"""Independent safetensors reader + elementwise verification against checkpoint EMA."""
import json
from pathlib import Path
from checkpoint import ROOT, DEFAULT_CHECKPOINT
import torch
from safetensors import safe_open
from export_weights import atomic_json, digest_file


def main():
    torch.set_num_threads(8)
    checkpoint=torch.load(DEFAULT_CHECKPOINT,map_location='cpu',weights_only=False,mmap=True)
    architecture=json.loads((ROOT/'fixtures/architecture.json').read_text())
    checkpoint_hash=digest_file(DEFAULT_CHECKPOINT)
    assert checkpoint_hash==architecture['checkpoint_sha256']
    results=[]
    for precision in ['fp32','fp16','int8']:
        base=ROOT/'weights'/precision
        manifest=json.loads((base/'manifest.json').read_text())
        assert manifest['checkpoint_sha256']==checkpoint_hash
        entries={}
        for name,entry in manifest['tensors'].items():
            entries.setdefault(entry['shard'],[]).append((name,entry))
        count=0
        for shard in manifest['shards']:
            path=base/shard['file']
            assert path.stat().st_size==shard['byteLength']<25_000_000
            assert digest_file(path)==shard['sha256']
            with safe_open(path,framework='pt',device='cpu') as f:
                assert f.metadata()=={'format':'rfd3-packed-v1','precision':precision}
                expected_keys=set()
                for name,entry in entries[shard['file']]:
                    source=checkpoint['model']['shadow.'+name].float()
                    tensor=f.get_tensor(name)
                    assert list(tensor.shape)==entry['shape']
                    expected_keys.add(name)
                    if entry['dtype']=='F16':
                        assert torch.equal(tensor,source.half()),name
                    elif entry['dtype']=='I8':
                        scales=source.abs().amax(dim=1)/127
                        scales=torch.where(scales==0,torch.ones_like(scales),scales)
                        assert torch.equal(f.get_tensor(entry['scale_key']),scales),name
                        assert torch.equal(tensor,torch.round(source/scales[:,None]).clamp(-127,127).to(torch.int8)),name
                        expected_keys.add(entry['scale_key'])
                    else:
                        assert torch.equal(tensor,source),name
                    count+=1
                assert set(f.keys())==expected_keys
        assert count==806
        results.append(dict(precision=precision,tensors=count,shards=len(manifest['shards']),total_bytes=manifest['total_bytes'],
                            all_shas_valid=True,elementwise_match=True,reader='safetensors 0.6.2'))
    atomic_json(ROOT/'fixtures/export-verification.json',dict(checkpoint_sha256=checkpoint_hash,results=results))
    print(json.dumps(results),flush=True)

if __name__=='__main__': main()
