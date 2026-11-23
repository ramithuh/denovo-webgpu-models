"""Independent Open Babel / libmolgrid / TorchScript references on crystallographic poses."""
import json
import re
import subprocess
import numpy as np
import torch
from openbabel import openbabel as ob
from build_grid_oracle import build
from download_weights import ROOT, UP, MODELS
from export_gnina import load, restore, reload_bundle

FIX = ROOT/"fixtures"
OUT = ROOT/".cache/fixtures"
RADII = {}
for line in (UP/"libmolgrid/src/atom_typer.cpp").read_text().splitlines():
    match = re.search(r'\{\s*\w+,\s*"(\w+)",\s*"\w+",\s*(\d+),([^}]+)\}', line)
    if match:
        RADII[match[1]] = float(match[3].split(",")[5])

def read(path):
    conv = ob.OBConversion()
    conv.SetInFormat(path.suffix[1:])
    mol = ob.OBMol()
    if not conv.ReadFile(mol,str(path)):
        raise ValueError(path)
    return mol

def prepare_ligand(case):
    mol = read(FIX/f"{case}_lig.sdf")
    mol.StripSalts()
    mol.AddHydrogens()
    mol.PerceiveBondOrders()
    mol.SetAromaticPerceived()
    mol.DeleteNonPolarHydrogens()
    return mol

def mapped(mol, lines, offset):
    mapping = {name:i+offset for i,line in enumerate(lines.splitlines()) for name in line.split()}
    atoms = []
    symbols = {"F":"Fluorine","Cl":"Chlorine","Br":"Bromine","I":"Iodine","Ca":"Calcium",
               "Zn":"Zinc","Mg":"Magnesium","Mn":"Manganese","Fe":"Iron","B":"Boron","P":"Phosphorus"}
    for atom in ob.OBMolAtomIter(mol):
        el = ob.GetSymbol(atom.GetAtomicNum())
        if el == "H":
            continue
        neighbors = [a.GetAtomicNum() for a in ob.OBAtomAtomIter(atom)]
        if el == "C":
            name = ("Aromatic" if atom.IsAromatic() else "Aliphatic")+"CarbonXS"
            name += "NonHydrophobe" if any(n not in (1,6) for n in neighbors) else "Hydrophobe"
        elif el in ("N","O"):
            name = "Nitrogen" if el=="N" else "Oxygen"
            suffix = ("Donor" if 1 in neighbors else "")+("Acceptor" if el=="O" or atom.IsHbondAcceptor() else "")
            if suffix:
                name += "XS"+suffix
        elif el=="S":
            name="SulfurAcceptor" if atom.IsHbondAcceptor() else "Sulfur"
        else:
            name = symbols.get(el,"GenericMetal")
        atoms.append([atom.GetX(),atom.GetY(),atom.GetZ(),mapping[name],RADII[name]])
    return atoms

def save_trace(path, a):
    a = np.asarray(a,dtype="<f4")
    file = OUT/(path+".bin")
    file.parent.mkdir(parents=True,exist_ok=True)
    a.tofile(file)
    return {"file":path+".bin","length":a.size,"shape":list(a.shape)}

def blocks(model, grid, name):
    x = grid
    if name.startswith("dense"):
        x = model.data_enc_init_conv(model.data_enc_init_pool(x)).relu()
        yield "stem", x
        for level in range(3):
            x = getattr(model, f"dense_block_{level}")(x)
            yield f"block{level}", x
            if level<2:
                x = getattr(model,f"data_enc_level{level}_bottleneck")(x).relu()
                yield f"bottleneck{level}", x
                x = getattr(model,f"data_enc_level{level+1}_pool")(x)
    else:
        for unit in range(1,6):
            if unit%2:
                x = getattr(model,f"avgpool_{unit//2}")(x)
            x = getattr(model,f"unit{unit}_conv")(x).relu()
            yield f"conv{unit}",x

@torch.inference_mode()
def main():
    gridder=build()
    torch.set_num_threads(3)
    OUT.mkdir(parents=True,exist_ok=True)
    _,meta=load(next(iter(MODELS)))
    bundles={p:reload_bundle(ROOT/"weights/default"/p) for p in ("fp32","fp16","int8")}
    summary={}
    for case in ("184l","10gs"):
        rec=read(FIX/f"{case}_rec.pdb")
        rec.AddHydrogens(True)
        lig=prepare_ligand(case)
        receptor=mapped(rec,meta["recmap"],0)
        ligand=mapped(lig,meta["ligmap"],14)
        coords=np.array([[a.GetX(),a.GetY(),a.GetZ()] for a in ob.OBMolAtomIter(lig)],dtype=np.float32)
        center=coords.sum(axis=0,dtype=np.float32)/len(coords)
        for pose in ("native","decoy"):
            key=f"{case}_{pose}"
            rows=np.array(ligand,dtype=np.float32)
            origin=center.copy()
            if pose=="decoy":
                # 90 degrees about z, followed by a fixed 9 A / 6 A shift.
                xy=rows[:,:2]-center[:2]
                rows[:,0]=center[0]-xy[:,1]+9
                rows[:,1]=center[1]+xy[:,0]+6
                origin[:2]+=[9,6]
            atoms=np.concatenate([receptor,rows])
            data=" ".join(map(str,origin.tolist()))+"\n"+"\n".join(" ".join(map(str,row)) for row in atoms)
            buf=subprocess.check_output([str(gridder)],input=data.encode())
            grid=torch.from_numpy(np.frombuffer(buf,"<f4").copy().reshape(1,28,48,48,48))
            trace={"center":origin.tolist(),"grid":save_trace(key+"/grid",grid),"models":{}}
            for p,weights in bundles.items():
                scores={}
                for name in MODELS:
                    model,_=load(name)
                    restore(model,name,weights)
                    logp,aff=model(grid)
                    score=logp.softmax(1)[0,1].item()
                    scores[name]={"cnnScore":score,"cnnAffinity":aff.item(),"cnnVS":score*aff.item()}
                    trace["models"][p+"/"+name]={b:save_trace(f"{key}/{p}/{name}/{b}",x)
                                                   for b,x in blocks(model,grid,name)}
                s=np.mean([v["cnnScore"] for v in scores.values()])
                a=np.mean([v["cnnAffinity"] for v in scores.values()])
                summary.setdefault(key,{})[p]={"cnnScore":s,"cnnAffinity":a,"cnnVS":s*a,"models":scores}
            (OUT/(key+".json")).write_text(json.dumps(trace,indent=2)+"\n")
            print(key,summary[key]["fp32"]["cnnScore"],flush=True)
    (FIX/"oracle_scores.json").write_text(json.dumps(summary,indent=2)+"\n")

if __name__=="__main__":
    main()
