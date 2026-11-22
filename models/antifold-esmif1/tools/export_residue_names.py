"""Pin Biotite's CCD peptide membership and three-to-one conversion table."""
import json
from pathlib import Path
from biotite.structure.info import amino_acid_names
from biotite.sequence import ProteinSequence
root=Path(__file__).resolve().parents[1]
names=sorted(amino_acid_names());mapping={}
for name in names:
    try:mapping[name]=ProteinSequence.convert_letter_3to1(name)
    except KeyError:pass
(root/'shared/residue-names.js').write_text('/** Peptide CCD names from pinned Biotite 1.5.0; matches its backbone filter. */\nexport const PEPTIDE_NAMES=new Set('+json.dumps(names,separators=(',',':'))+');\nexport const RESIDUE_LETTERS='+json.dumps(mapping,separators=(',',':'))+';\n')
