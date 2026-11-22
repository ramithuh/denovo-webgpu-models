"""Exercise the actual exporter with a deliberately tiny shard cap."""
import tempfile
import unittest
import json
import torch
from reference import ROOT
from export_thermompnn import export,reload_bundle
class ExportTest(unittest.TestCase):
    def test_multipart_and_colocated_scales(self):
        torch.manual_seed(123)
        original={'matrix.weight':torch.randn(200,100), 'vector.bias':torch.randn(200),
                  'prot_mpnn.W_s.weight':torch.randn(21,128)}
        for precision in ['fp32','fp16','int8']:
            with tempfile.TemporaryDirectory(dir=ROOT/'.cache/tmp') as temporary:
                from pathlib import Path
                path=Path(temporary);m=export(original,precision,path,cap=8192)
                self.assertTrue(any('parts' in t for t in m['tensors'].values()))
                self.assertTrue(all(s['byteLength']<=8192 for s in m['shards']))
                reloaded=reload_bundle(path)
                torch.testing.assert_close(original['vector.bias'],reloaded['vector.bias'],rtol=0,atol=0)
                torch.testing.assert_close(original['prot_mpnn.W_s.weight'],reloaded['prot_mpnn.W_s.weight'],rtol=0,atol=0)
                self.assertLess(float((original['matrix.weight']-reloaded['matrix.weight']).abs().max()),0.02)
if __name__=='__main__':unittest.main()
