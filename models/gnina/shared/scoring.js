export function centroid(pose) {
  const atoms=pose.centerAtoms??pose.atoms;
  if(!atoms.length) throw Error("Empty pose");
  return ["x","y","z"].map(k=>Math.fround(atoms.reduce((s,a)=>Math.fround(s+Math.fround(a[k])),0)/atoms.length));
}
export function transformPose(pose,rotation=[0,0,0],translation=[0,0,0]) {
  const [x,y,z]=centroid(pose);
  const [cx,cy,cz]=rotation.map(Math.cos),[sx,sy,sz]=rotation.map(Math.sin);
  function move(a) {
    const dx=a.x-x,dy=a.y-y,dz=a.z-z, ry=dy*cx-dz*sx,rz=dy*sx+dz*cx;
    const rx=dx*cy+rz*sy;
    return {...a,x:x+rx*cz-ry*sz+translation[0],y:y+rx*sz+ry*cz+translation[1],z:z-dx*sy+rz*cy+translation[2]};
  }
  return {...pose,atoms:pose.atoms.map(move),centerAtoms:(pose.centerAtoms??pose.atoms).map(move)};
}
export function prepareGrid(rec,lig,center=centroid(lig)) {
  if(!Array.isArray(rec?.atoms)||!Array.isArray(lig?.atoms)||center.length!==3||!center.every(Number.isFinite)) throw Error("Invalid complex");
  const origin=center.map(v=>Math.fround(v-11.75)),groups=Array.from({length:28},()=>[]);
  for(const molecule of [rec,lig]) for(const a of molecule.atoms) {
    if(![a.x,a.y,a.z,a.radius].every(Number.isFinite)||a.radius<=0||!Number.isInteger(a.channel)||!groups[a.channel]) throw Error("Invalid typed atom");
    const r=1.5*a.radius;
    if([a.x,a.y,a.z].every((v,i)=>v+r>=origin[i]&&v-r<=origin[i]+23.5)) groups[a.channel].push(a);
  }
  return {origin,groups};
}
export function gridCPU({origin,groups}) {
  const out=new Float32Array(28*48**3);
  groups.forEach((group,c)=>{
    for(const a of group) {
      const coord=[a.x,a.y,a.z].map(Math.fround), r=Math.fround(a.radius);
      const lo=coord.map((v,i)=>Math.max(0,Math.floor((v-1.5*r-origin[i])*2)));
      const hi=coord.map((v,i)=>Math.min(48,Math.ceil((v+1.5*r-origin[i])*2)));
      for(let x=lo[0];x<hi[0];x++) for(let y=lo[1];y<hi[1];y++) {
        const xy=(origin[0]+x*.5-coord[0])**2+(origin[1]+y*.5-coord[1])**2;
        for(let z=lo[2];z<hi[2];z++) {
          const d=Math.sqrt(xy+(origin[2]+z*.5-coord[2])**2)/r;
          if(d<1.5) out[((c*48+x)*48+y)*48+z]+=d<=1?Math.exp(-2*d*d):4*Math.exp(-2)*(d-1.5)**2;
        }
      }
    }
  });
  return out;
}
const clamp01=x=>Math.max(0,Math.min(1,x));
// Surface distance d is the center distance minus the two XS radii.
export function vinaPairEnergy(l,r,d,terms) {
  const g1=-.035579*Math.exp(-4*d*d),g2=-.005156*Math.exp(-((d-3)**2)/4);
  const rep=.840245*Math.min(d,0)**2;
  const hydro=r.hydrophobic&&l.hydrophobic?-.035069*clamp01(1.5-d):0;
  const hb=(r.acceptor&&l.donor)||(l.acceptor&&r.donor)?-.587439*clamp01(-d/.7):0;
  if(terms) {
    terms.gauss1+=g1;terms.gauss2+=g2;terms.repulsion+=rep;terms.hydrophobic+=hydro;terms.hbond+=hb;
  }
  return g1+g2+rep+hydro+hb;
}
export function vinaBreakdown(rec,lig) {
  const terms={gauss1:0,gauss2:0,repulsion:0,hydrophobic:0,hbond:0}, perAtom=new Float64Array(lig.atoms.length), perResidue=new Map();
  const penalty=1+.05846*lig.rotors;
  for(const [i,l] of lig.atoms.entries()) for(const r of rec.atoms) {
    const rsq=(r.x-l.x)**2+(r.y-l.y)**2+(r.z-l.z)**2;
    if(rsq>=64) continue;
    const d=Math.sqrt(rsq)-r.radius-l.radius;
    const energy=vinaPairEnergy(l,r,d,terms)/penalty;
    perAtom[i]+=energy;perResidue.set(r.residue,(perResidue.get(r.residue)??0)+energy);
  }
  return {vinaScore:Object.values(terms).reduce((a,b)=>a+b)/penalty,terms,penalty,perAtom,perResidue:Object.fromEntries(perResidue)};
}
export const vinaScore=(rec,lig)=>vinaBreakdown(rec,lig).vinaScore;
