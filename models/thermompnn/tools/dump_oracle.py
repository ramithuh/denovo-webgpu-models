import json
from reference import ROOT, load_model, restore, run, COMMIT
from export_thermompnn import reload_bundle
import ast
import torch
from protein_mpnn_utils import parse_PDB
if __name__=='__main__':
    cases=json.loads((ROOT/'fixtures/cases.json').read_text());cases.append({'id':'full_length','pdb':'full.pdb','input':'full.pdb','benchmark':True});model=load_model()
    for precision in ['fp32','fp16','int8']:
        restore(model,reload_bundle(ROOT/'weights/default'/precision))
        for case in cases:
            if 'same_as' in case:continue
            traces=run(model,case)
            if case.get('benchmark'):traces={k:traces[k] for k in ['S','mask','ddg','targeted']}
            meta={'source':'unchanged upstream TransferModel.forward via hooks','upstream_commit':COMMIT,
                  'precision':precision,'case':case,'tensors':{}}
            binary=bytearray()
            for name,a in traces.items():
                meta['tensors'][name]={'shape':list(a.shape),'dtype':'I32' if a.dtype.kind=='i' else 'F32','offset':len(binary),'byteLength':a.nbytes}
                binary.extend(a.tobytes())
            path=ROOT/'fixtures'/f'oracle_{case["id"]}_{precision}'
            path.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n');path.with_suffix('.bin').write_bytes(binary)
            print(path.name,len(binary),flush=True)

    # Execute the exact upstream analysis helper without importing its unrelated
    # training/benchmark dependencies. The function AST and code are unchanged.
    source=ROOT/'analysis/thermompnn_benchmarking.py'
    function=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='compute_centrality')
    namespace={'torch':torch}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
    centralities={}
    for case in cases:
        if case.get('same_as') or case.get('benchmark'):continue
        pdb=parse_PDB(str(ROOT/'fixtures'/case['pdb']),case.get('chains'))[0]
        values=[]
        for key in pdb:
            if key.startswith('seq_chain_'):
                chain=key[len('seq_chain_'):]
                values.extend(namespace['compute_centrality'](pdb['coords_chain_'+chain],chain=chain).tolist())
        centralities[case['id']]=values
    (ROOT/'fixtures/centrality.json').write_text(json.dumps(centralities,indent=2)+'\n')
