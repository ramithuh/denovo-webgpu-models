import {polarHydrogens} from "./hydrogen.js";
// Distances in angstroms. XS radii are shared by the Vina terms and libmolgrid.
const xs = {C:1.9,N:1.8,O:1.7,S:2,P:2.1,F:1.5,Cl:1.8,Br:2,I:2.2,B:1.92};
const covalent = {H:.31,C:.76,N:.71,O:.66,S:1.05,P:1.07,F:.57,Cl:1.02,Br:1.2,I:1.39,B:.85};
const metals = new Set("Li Na K Rb Cs Be Mg Ca Sr Ba Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Mo Ru Rh Pd Ag Cd Sn W Re Os Ir Pt Au Hg Al Ga In Bi".split(" "));
const aromaticSidechains = {
  PHE:"CG CD1 CD2 CE1 CE2 CZ",TYR:"CG CD1 CD2 CE1 CE2 CZ",
  TRP:"CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2", HIS:"CG ND1 CD2 CE1 NE2",
};
function element(value) {
  const s = String(value).trim().toLowerCase();
  return s === "d" ? "H" : s.charAt(0).toUpperCase()+s.slice(1);
}
function parseCharge(text) {
  const value=text.trim().replace(/^(\d)([+-])$/,"$2$1");
  return Number(value)||0;
}
function molecule(atoms, bonds = []) {
  if (!atoms.length || atoms.some(a=>![a.x,a.y,a.z].every(Number.isFinite))) throw Error("Invalid or empty coordinates");
  return {atoms,bonds};
}
function inferBonds(atoms, existing) {
  const bonds = existing.slice(), seen = new Set(bonds.map(([a,b])=>[Math.min(a,b),Math.max(a,b)].join("/")));
  const cells = new Map();
  for (let i=0;i<atoms.length;i++) {
    const a=atoms[i], r=covalent[a.el];
    if (!r) continue;
    const cell=[a.x,a.y,a.z].map(v=>Math.floor(v/3.4));
    for(let x=-1;x<=1;x++) for(let y=-1;y<=1;y++) for(let z=-1;z<=1;z++) {
      for(const j of cells.get([cell[0]+x,cell[1]+y,cell[2]+z].join())??[]) {
        const b=atoms[j], limit=r+covalent[b.el]+.42;
        const d=(a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2;
        if(d>.16 && d<limit*limit && !(a.el==="H"&&b.el==="H") && !seen.has(j+"/"+i)) bonds.push([j,i,1]);
      }
    }
    const key=cell.join();
    if(!cells.has(key)) cells.set(key,[]);
    cells.get(key).push(i);
  }
  return bonds;
}
function pdbElement(name,raw) {
  if(raw.trim()) return element(raw);
  if(name[0]===" " || /^\d/.test(name)) return element(name.replace(/[\d ]/g,"")[0]);
  const pair=element(name.slice(0,2));
  return pair in covalent || metals.has(pair) ? pair : element(name[0]);
}
function selectAtom(a,atoms,ids) {
  const id=[a.chain,a.seq,a.ins,a.res,a.name].join("/");
  if(ids.has(id)) {
    const i=ids.get(id);
    if(a.occ>atoms[i].occ || (a.occ===atoms[i].occ && a.alt<atoms[i].alt)) atoms[i]=a;
  } else {ids.set(id,atoms.length);atoms.push(a);}
}
export function parsePDB(text) {
  const atoms=[],ids=new Map(),links=[];
  for(const l of text.split(/\r?\n/)) {
    if(l.startsWith("ENDMDL")) break;
    if(l.startsWith("CONECT")) {
      const serials=l.slice(6).match(/.{1,5}/g)?.map(Number)??[];
      for(const j of serials.slice(1)) links.push([serials[0],j]);
    }
    if(!/^(ATOM  |HETATM)/.test(l)) continue;
    const name=l.slice(12,16);
    selectAtom({el:pdbElement(name,l.slice(76,78)), name:name.trim(),alt:l[16].trim(),res:l.slice(17,20).trim(),
      charge:parseCharge(l.slice(78,80)),chain:l[21],seq:l.slice(22,26).trim(),ins:l[26].trim(),serial:Number(l.slice(6,11)),occ:Number(l.slice(54,60)),
      x:parseFloat(l.slice(30,38)),y:parseFloat(l.slice(38,46)),z:parseFloat(l.slice(46,54))},atoms,ids);
  }
  const serials=new Map(atoms.map((a,i)=>[a.serial,i])), unique=new Map();
  for (const [s,t] of links) if(serials.has(s)&&serials.has(t)&&s!==t) {
    const pair=[serials.get(s),serials.get(t)].sort((a,b)=>a-b);
    unique.set(pair.join(),[...pair,1]);
  }
  return molecule(atoms,inferBonds(atoms,[...unique.values()]));
}
export function parseMMCIF(text) {
  const tokens=text.match(/^;[^\n]*\n[\s\S]*?^;[^\n]*|#[^\n]*|'[^'\n]*'(?=\s|$)|"[^"\n]*"(?=\s|$)|[^\s]+/gm)?.filter(t=>t[0]!=="#")??[];
  const atoms=[],ids=new Map();
  let model;
  for(let i=0;i<tokens.length;i++) {
    if(tokens[i].toLowerCase()!=="loop_") continue;
    const fields=[];
    while(tokens[i+1]?.startsWith("_")) fields.push(tokens[++i].toLowerCase());
    if(!fields.includes("_atom_site.cartn_x")) continue;
    const k=Object.fromEntries(fields.map((f,n)=>[f.slice(11),n]));
    for(i++;i<tokens.length && !/^(_|loop_|stop_|data_|save_)/i.test(tokens[i]);i+=fields.length) {
      const get=(...names)=>{
        for(const name of names) if(k[name]!==undefined) {
          const t=tokens[i+k[name]]?.replace(/^(['"])(.*)\1$/,"$2");
          if(t && t!=="." && t!=="?") return t;
        }
        return "";
      };
      const m=get("pdbx_pdb_model_num");
      model??=m;
      if(m!==model) continue;
      selectAtom({el:element(get("type_symbol")),name:get("auth_atom_id","label_atom_id"),alt:get("label_alt_id"),
        res:get("auth_comp_id","label_comp_id"),charge:parseCharge(get("pdbx_formal_charge")),chain:get("auth_asym_id","label_asym_id"),
        seq:get("auth_seq_id","label_seq_id"),ins:get("pdbx_pdb_ins_code"),occ:Number(get("occupancy")),
        x:parseFloat(get("cartn_x")),y:parseFloat(get("cartn_y")),z:parseFloat(get("cartn_z"))},atoms,ids);
    }
    i--;
  }
  return molecule(atoms,inferBonds(atoms,[]));
}
function sdfRecord(text) {
  const lines=text.replace(/^\r?\n(?=[^\n]*\n[^\n]*\n[^\n]*\n[^\n]*V[23]000)/,"").split(/\r?\n/);
  const counts=lines.findIndex(l=>/V[23]000\s*$/.test(l));
  if(counts<0) throw Error("Missing SDF counts line");
  const atoms=[],bonds=[];
  if(lines[counts].includes("V3000")) {
    const ids=new Map(); let section="";
    for(const l of lines) {
      if(!l.startsWith("M  V30 ")) continue;
      const [id,type,x,y,z,...rest]=l.slice(7).trim().split(/\s+/);
      if(id==="BEGIN"||id==="END") {section=id==="BEGIN"?type:"";continue;}
      if(section==="ATOM") {
        ids.set(id,atoms.length);atoms.push({el:element(type),x:+x,y:+y,z:+z,charge:+(rest.find(s=>s.startsWith("CHG="))?.slice(4)??0)});
      } else if(section==="BOND") bonds.push([ids.get(x),ids.get(y),+type]);
    }
  } else {
    const n=Number(lines[counts].slice(0,3)),m=Number(lines[counts].slice(3,6));
    for(const l of lines.slice(counts+1,counts+1+n)) {
      const code=Number(l.slice(36,39));
      atoms.push({el:element(l.slice(31,34)),x:parseFloat(l.slice(0,10)),y:parseFloat(l.slice(10,20)),z:parseFloat(l.slice(20,30)),charge:code&&code!==4?4-code:0});
    }
    for(const l of lines.slice(counts+1+n,counts+1+n+m)) bonds.push([+l.slice(0,3)-1,+l.slice(3,6)-1,+l.slice(6,9)]);
    for(const l of lines) if(l.startsWith("M  CHG")) {
      const nums=l.trim().split(/\s+/).slice(3).map(Number);
      for(let j=0;j<nums.length;j+=2) atoms[nums[j]-1].charge=nums[j+1];
    }
  }
  return molecule(atoms,bonds);
}
export function parseSDF(text) {return text.split(/^\$\$\$\$\s*$/m).filter(r=>r.trim()).map(sdfRecord);}
export function parseMOL2(text) {
  const atoms=[],bonds=[],ids=new Map();
  let section="";
  for(const line of text.split(/\r?\n/)) {
    const l=line.trim();
    if(l.startsWith("@<TRIPOS>")) {section=l.slice(9);if(section==="MOLECULE"&&atoms.length) break;continue;}
    if(!l || l.startsWith("#")) continue;
    const [id,a,b,c,d,type]=l.split(/\s+/);
    if(section==="ATOM") {
      ids.set(id,atoms.length);
      atoms.push({el:element(type.split(".")[0]),sybyl:type,aromatic:type.endsWith(".ar"),x:+b,y:+c,z:+d});
    } else if(section==="BOND") bonds.push([ids.get(a),ids.get(b),c==="ar"?4:c==="am"?1:Number(c)]);
  }
  return molecule(atoms,bonds);
}
function aromaticRings(atoms,adj) {
  const aromatic=new Set();
  // Enumerate small conjugated cycles only. Kekule SDF aromatic rings have
  // alternating double bonds (a lone-pair heteroatom can complete a five-ring).
  for(let start=0;start<atoms.length;start++) {
    if(!adj[start].some(b=>b.order>=2)) continue;
    function visit(path,orders) {
      const i=path.at(-1);
      for(const {j,order} of adj[i]) {
        if(j===start && path.length>=5) {
          const edges=[...orders,order], doubles=edges.filter(o=>o===2).length;
          if(edges.every(o=>o===4) || (doubles===3&&edges.length===6&&edges.every((o,k)=>o!==edges[(k+1)%6])) ||
              (doubles===2 && edges.length===5 && path.some(k=>["N","O","S"].includes(atoms[k].el)) &&
               edges.every((o,k)=>o!==2 || edges[(k+1)%5]!==2))) for(const atom of path) aromatic.add(atom);
        } else if(path.length<6 && j>start && !path.includes(j)&&["C","N","S","O"].includes(atoms[j].el)) visit([...path,j],[...orders,order]);
      }
    }
    visit([start],[]);
  }
  return aromatic;
}
function bridges(adj) {
  const result=new Set(),time=new Int32Array(adj.length),low=new Int32Array(adj.length);
  let next=0;
  function walk(i,parent) {
    low[i]=time[i]=++next;
    for(const {j,edge} of adj[i]) {
      if(edge===parent) continue;
      if(!time[j]) {
        walk(j,edge);low[i]=Math.min(low[i],low[j]);
        if(low[j]>time[i]) result.add(edge);
      } else low[i]=Math.min(low[i],time[j]);
    }
  }
  for(let i=0;i<adj.length;i++) if(!time[i]) walk(i,-1);
  return result;
}
export function typeMolecule({atoms,bonds}, role) {
  if(!["ligand","receptor"].includes(role)) throw Error("Unknown atom typing role");
  const lig=role==="ligand", adj=atoms.map(()=>[]);
  bonds.forEach(([i,j,order],edge)=>{
    if(!adj[i]||!adj[j]||i===j||![1,2,3,4].includes(order)) throw Error("Invalid bond");
    adj[i].push({j,order,edge});adj[j].push({j:i,order,edge});
  });
  const aromatic=aromaticRings(atoms,adj), typed=[],centerAtoms=[],addedHydrogens=[];
  for(let i=0;i<atoms.length;i++) {
    const a=atoms[i], ns=adj[i].map(({j})=>atoms[j]), el=a.el==="Se"?"S":a.el;
    if(el==="H") {if(ns.some(b=>b.el!=="C")) centerAtoms.push(a);continue;}
    centerAtoms.push(a);
    const hetero=ns.some(b=>b.el!=="H"&&b.el!=="C");
    const ar=aromatic.has(i)||a.aromatic||aromaticSidechains[a.res]?.split(" ").includes(a.name);
    let donor=ns.some(b=>b.el==="H"), acceptor=false, hydrophobic=false;
    const valence=adj[i].reduce((s,b)=>s+(b.order===4?1.5:b.order),0);
    if(lig && !a.res && (el==="O"||el==="N")) {
      const charge=a.charge??0;
      const target=el==="O"?2+charge:a.sybyl==="N.4"?4:3+charge;
      const missing=Math.max(0,Math.floor(target-valence+1e-6));
      if(missing && a.sybyl!=="O.co2"){
        donor=true;
        const parent=adj[i][0]?.j;
        const others=parent===undefined?[]:adj[parent].filter(b=>b.j!==i);
        const carbonyl=others.find(b=>b.order===2&&atoms[b.j].el==="O");
        const ref=carbonyl??others.at(-1);
        const conjugated=ns.some((n,k)=>n.el==="C" && adj[adj[i][k].j].some(b=>b.order===2&&atoms[b.j].el==="O"));
        const planar=!!ar||conjugated||adj[i].some(b=>b.order>=2)||["N.am","N.pl3","N.2","O.2"].includes(a.sybyl);
        addedHydrogens.push(...polarHydrogens(a,ns,missing,planar,ref?atoms[ref.j]:undefined));
      }
    } else if(!donor && a.res && !ns.some(b=>b.el==="H")) {
      // Canonical residue donor inference for structures deposited without hydrogens.
      if(el==="N") donor=(a.name==="N"&&(a.res!=="PRO"||ns.length<3)) || ["NZ","NH1","NH2","NE","NE1","ND2","NE2"].includes(a.name);
      if(el==="O") {
        donor=["OG","OG1","OH"].includes(a.name);
        if(ns.length===1 && ns[0].el==="C") {
          const oxy=adj[adj[i][0].j].map(b=>b.j).filter(j=>atoms[j].el==="O");
          // Open Babel perceives the carbonyl first in its z-sorted bond list.
          if(oxy.length===2 && oxy.every(j=>adj[j].length===1) && (a.charge??0)===0)
            donor=oxy.sort((j,k)=>atoms[j].z-atoms[k].z)[1]===i;
        }
      }
    }
    if(el==="O") acceptor=true;
    if(el==="N") {
      const amide=adj[i].some(({j})=>atoms[j].el==="C"&&adj[j].some(b=>b.order===2&&atoms[b.j].el==="O"));
      acceptor=(a.charge??0)<=0 && ns.length<4 && !amide && !(ar&&donor) &&
        !["N.am","N.4","N.pl3"].includes(a.sybyl) && (!a.res || a.res==="HIS" || (a.name==="N" && ns.filter(b=>b.el!=="H").length===1));
    }
    let ch;
    if(el==="C") {ch=(ar?2:0)+(hetero?1:0);hydrophobic=!hetero;}
    else if(el==="N") ch=(lig?7:5)+Number(donor);
    else if(el==="O") ch=(lig?9:7)+Number(donor);
    else if(el==="S") {ch=lig?11:9;donor=false;acceptor=false;}
    else if(el==="P") ch=lig?12:10;
    else if(["F","Cl","Br","I"].includes(el)) {ch=lig?({F:6,Cl:5,Br:4,I:4})[el]:4;hydrophobic=true;}
    else if(/^[A-Z][a-z]?$/.test(el)) {ch=lig?13:el==="Ca"?11:el==="Zn"?12:13;donor=el!=="B";hydrophobic=el==="B";}
    else throw Error("Unsupported element "+el);
    typed.push({...a,index:i,channel:ch+(lig?14:0),radius:xs[el]??1.2,donor,acceptor,hydrophobic,
      residue:[a.chain,a.seq,a.ins,a.res].filter(Boolean).join("/")});
  }
  let rotors=0;
  if(lig) {
    const cut=bridges(adj);
    for(const index of cut) {
      const [i,j,order]=bonds[index];
      if(order!==1 || [i,j].some(k=>adj[k].filter(b=>atoms[b.j].el!=="H").length<2)) continue;
      const amide=[[i,j],[j,i]].some(([n,c])=>atoms[n].el==="N" && atoms[c].el==="C" && adj[c].some(b=>b.order===2&&atoms[b.j].el==="O"));
      if(!amide) rotors++;
    }
  }
  if(!typed.length) throw Error("No heavy atoms");
  return {atoms:typed,centerAtoms:centerAtoms.concat(addedHydrogens),rotors};
}
export function parseReceptor(text,format="pdb") {
  return typeMolecule(/cif/i.test(format)?parseMMCIF(text):parsePDB(text),"receptor");
}
export function parseLigands(text,format="sdf") {
  const poses=/mol2/i.test(format)?text.split(/(?=@<TRIPOS>MOLECULE)/).filter(s=>s.trim()).map(parseMOL2):parseSDF(text);
  if(!poses.length) throw Error("No ligand poses");
  return poses.map(m=>typeMolecule(m,"ligand"));
}
