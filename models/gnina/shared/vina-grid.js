import {vinaPairEnergy} from "./scoring.js";
import {checkAbort} from "./loader.js";

const axes=["x","y","z"];
const clamp=(x,lo,hi)=>Math.min(hi,Math.max(lo,x));
export const yieldTask=()=>new Promise(resolve=>setTimeout(resolve,0));

function validateBox(box) {
  if(![box?.center,box?.size].every(v=>v?.length===3 && v.every(Number.isFinite)) ||
      box.size.some(v=>v<=0 || v>64)) throw Error("Box requires finite center and side lengths in (0,64] Å");
  return {center:Array.from(box.center),size:Array.from(box.size)};
}

export function dockingBox(rec,lig,options={}) {
  if(options.box || options.center || options.size) return validateBox(options.box??options);
  const {autoboxAdd=4,pocketResidues,autoboxLigand}=options;
  if(!Number.isFinite(autoboxAdd)||autoboxAdd<0) throw Error("Invalid autoboxAdd");
  let atoms=(autoboxLigand??lig)?.atoms;
  if(pocketResidues) {
    const selected=new Set(pocketResidues);
    atoms=rec.atoms.filter(a=>selected.has(a.residue));
    if(!atoms.length) throw Error("No atoms in selected pocket residues");
  }
  if(!atoms?.length) throw Error("Specify a docking box, autoboxLigand, or pocketResidues");
  const lo=[Infinity,Infinity,Infinity],hi=lo.map(v=>-v);
  for(const a of atoms) for(let d=0;d<3;d++) {
    lo[d]=Math.min(lo[d],a[axes[d]]);hi[d]=Math.max(hi[d],a[axes[d]]);
  }
  return validateBox({center:lo.map((v,d)=>(v+hi[d])/2),size:lo.map((v,d)=>clamp(hi[d]-v+2*autoboxAdd,4,64))});
}

// Tabulate the same five radial terms as vinaBreakdown. The inner stencil
// interpolates in squared distance, avoiding exponentials per lattice node.
export function pairTable(l,r) {
  const table=new Float32Array(64*256+1);
  for(let i=0;i<table.length;i++) table[i]=vinaPairEnergy(l,r,Math.sqrt(i/256)-l.radius-r.radius);
  return table;
}
export const typeKey=a=>[a.radius,+a.hydrophobic,+a.acceptor,+a.donor].join("/");

export async function buildVinaGrids(rec,lig,box,{spacing=.375,signal,onProgress,check=()=>checkAbort(signal)}={}) {
  box=validateBox(box);
  if(!Number.isFinite(spacing)||spacing<.375||spacing>.5) throw Error("Grid spacing must be 0.375–0.5 Å");
  // Center a uniform lattice around the physical box. The harmonic wall starts
  // at the requested boundary even when the last cell extends beyond it.
  const cells=box.size.map(v=>Math.ceil(v/spacing)),step=[spacing,spacing,spacing];
  const origin=box.center.map((v,d)=>v-cells[d]*spacing/2),[nx,ny,nz]=cells.map(n=>n+1);
  const types=new Map(lig.atoms.map(a=>[typeKey(a),a])),keys=[...types.keys()];
  if(nx*ny*nz*4*types.size>256*1024**2) throw Error("Affinity grids exceed 256 MiB; reduce the docking box");
  const grids=[],atomTypes=Int32Array.from(lig.atoms.map(a=>keys.indexOf(typeKey(a))));
  const nearby=rec.atoms.filter(a=>axes.every((k,d)=>a[k]>=origin[d]-8 && a[k]<=origin[d]+cells[d]*spacing+8));
  let last=performance.now();
  onProgress?.({phase:"grid",completed:0,total:types.size});check();
  for(const l of types.values()){
    const values=new Float32Array(nx*ny*nz),tables=new Map();
    for(const r of nearby) {
      const key=typeKey(r);
      if(!tables.has(key)) tables.set(key,pairTable(l,r));
      const table=tables.get(key), dx=origin[0]-r.x,dy=origin[1]-r.y,dz=origin[2]-r.z;
      const x0=Math.max(0,Math.ceil((-8-dx)/step[0])),x1=Math.min(nx-1,Math.floor((8-dx)/step[0]));
      for(let x=x0;x<=x1;x++) {
        const xx=(dx+x*step[0])**2,ry=Math.sqrt(Math.max(0,64-xx));
        const y0=Math.max(0,Math.ceil((-ry-dy)/step[1])),y1=Math.min(ny-1,Math.floor((ry-dy)/step[1]));
        for(let y=y0;y<=y1;y++) {
          const xy=xx+(dy+y*step[1])**2,rz=Math.sqrt(Math.max(0,64-xy));
          const z0=Math.max(0,Math.ceil((-rz-dz)/step[2])),z1=Math.min(nz-1,Math.floor((rz-dz)/step[2]));
          const row=(x*ny+y)*nz;
          for(let z=z0;z<=z1;z++) {
            const s=(xy+(dz+z*step[2])**2)*256,i=s|0;
            if(i<16384) values[row+z]+=table[i]+(s-i)*(table[i+1]-table[i]);
          }
        }
      }
      if(performance.now()-last>40) {await yieldTask();check();last=performance.now();}
    }
    grids.push(values);onProgress?.({phase:"grid",completed:grids.length,total:types.size});check();
  }
  const lo=box.center.map((v,d)=>v-box.size[d]/2),hi=box.center.map((v,d)=>v+box.size[d]/2);
  return {origin,cells,lo,hi,invStep:step.map(v=>1/v),grids,atomTypes,box};
}

// Raw intermolecular energy, before the conformation-independent rotor penalty.
export function sampleVinaGrid(grid,p,g,cap=Infinity) {
  const {cells,invStep,origin,lo,hi,grids,atomTypes}=grid;
  const nz=cells[2]+1,yz=(cells[1]+1)*nz;
  let energy=0;
  for(let a=0;a<atomTypes.length;a++) {
    const i=3*a,data=grids[atomTypes[a]];
    const x=(p[i]-origin[0])*invStep[0],y=(p[i+1]-origin[1])*invStep[1],z=(p[i+2]-origin[2])*invStep[2];
    const cx=clamp(x,0,cells[0]),cy=clamp(y,0,cells[1]),cz=clamp(z,0,cells[2]);
    const ix=Math.min(cx|0,cells[0]-1),iy=Math.min(cy|0,cells[1]-1),iz=Math.min(cz|0,cells[2]-1);
    const fx=cx-ix,fy=cy-iy,fz=cz-iz,base=ix*yz+iy*nz+iz;
    const a00=data[base],a01=data[base+nz],a10=data[base+yz],a11=data[base+yz+nz];
    const d00=data[base+1]-a00,d01=data[base+nz+1]-a01,d10=data[base+yz+1]-a10,d11=data[base+yz+nz+1]-a11;
    const v00=a00+fz*d00,v01=a01+fz*d01,v10=a10+fz*d10,v11=a11+fz*d11;
    const v0=v00+fy*(v01-v00),v1=v10+fy*(v11-v10);
    const wx=p[i]-clamp(p[i],lo[0],hi[0]),wy=p[i+1]-clamp(p[i+1],lo[1],hi[1]),
      wz=p[i+2]-clamp(p[i+2],lo[2],hi[2]);
    const e=v0+fx*(v1-v0),scale=e>0?1/(1+e/cap):1,ds=scale*scale;
    energy+=scale*e+10*(wx*wx+wy*wy+wz*wz);
    g[i]=20*wx+ds*(x===cx?(v1-v0)*invStep[0]:0);
    g[i+1]=20*wy+ds*(y===cy?((1-fx)*(v01-v00)+fx*(v11-v10))*invStep[1]:0);
    g[i+2]=20*wz+ds*(z===cz?((1-fx)*((1-fy)*d00+fy*d01)+fx*((1-fy)*d10+fy*d11))*invStep[2]:0);
  }
  return energy;
}

// Off-lattice pairwise refinement of the retained minima. Restrict receptor
// neighbors to a sphere around each starting atom and rebuild its list after
// it moves through the 2 Å skin, including on rejected line-search trials.
export function pairwiseObjective(rec,tree,box) {
  const {atoms}=tree.ligand,lists=atoms.map(()=>[]),anchors=new Float64Array(3*atoms.length).fill(Infinity);
  const types=new Map(rec.atoms.map(a=>[typeKey(a),a])),keys=[...types.keys()];
  const receptorTypes=Int32Array.from(rec.atoms,a=>keys.indexOf(typeKey(a)));
  const ligandTables=new Map(atoms.map(a=>[typeKey(a),a]));
  for(const [key,l] of ligandTables) ligandTables.set(key,Array.from(types.values(),r=>pairTable(l,r)));
  const pairTables=atoms.map(a=>ligandTables.get(typeKey(a)));
  return (p,g)=>{
    let energy=0;
    for(let a=0;a<atoms.length;a++) {
      const i=3*a,x=p[i],y=p[i+1],z=p[i+2],typed=pairTables[a];
      if((x-anchors[i])**2+(y-anchors[i+1])**2+(z-anchors[i+2])**2>=4){
        anchors[i]=x;anchors[i+1]=y;anchors[i+2]=z;
        lists[a].length=0;
        for(let j=0;j<rec.atoms.length;j++){
          const r=rec.atoms[j];
          if((x-r.x)**2+(y-r.y)**2+(z-r.z)**2<100) lists[a].push(j);
        }
      }
      let gx=0,gy=0,gz=0;
      for(const j of lists[a]) {
        const r=rec.atoms[j],dx=x-r.x,dy=y-r.y,dz=z-r.z,s=(dx*dx+dy*dy+dz*dz)*256;
        if(s>=16384) continue;
        const k=s|0,table=typed[receptorTypes[j]],slope=table[k+1]-table[k],w=512*slope;
        energy+=table[k]+(s-k)*slope;
        gx+=w*dx;gy+=w*dy;gz+=w*dz;
      }
      g[i]=gx;g[i+1]=gy;g[i+2]=gz;
      for(let d=0;d<3;d++){
        const offset=p[i+d]-box.center[d],excess=offset-clamp(offset,-box.size[d]/2,box.size[d]/2);
        energy+=10*excess*excess;g[i+d]+=20*excess;
      }
    }
    return energy;
  };
}
