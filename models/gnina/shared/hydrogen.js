// Complete missing polar hydrogens for the ligand center. Hydrogen density
// channels are absent from all three default models.
const xyz=a=>[a.x,a.y,a.z];
const sub=(a,b)=>a.map((x,i)=>x-b[i]);
const add=(a,b)=>a.map((x,i)=>x+b[i]);
const scale=(a,k)=>a.map(x=>x*k);
const norm=a=>scale(a,1/(Math.hypot(...a)||1));
const cross=([a,b,c],[d,e,f])=>[b*f-c*e,c*d-a*f,a*e-b*d];

export function polarHydrogens(atom,neighbors,count,planar,reference) {
  const center=xyz(atom), bonds=neighbors.map(n=>norm(sub(center,xyz(n))));
  const radius=atom.el==="O"?.66:atom.el==="N"?.71:1.05;
  const length=.31+radius*(planar?.95:1), out=[];
  while(out.length<count) {
    let dir;
    if(!bonds.length) dir=[1,0,0];
    else if(bonds.length===1) {
      const axis=bonds[0];
      const ref=reference?sub(xyz(neighbors[0]),xyz(reference)):[0,0,0];
      let normal=cross(axis,ref);
      if(Math.hypot(...normal)<1e-9) normal=cross(axis,Math.abs(axis[0])<.8?[1,0,0]:[0,1,0]);
      const side=norm(cross(axis,normal));
      dir=sub(axis,scale(side,planar?Math.sqrt(3):Math.tan((180-109.471)*Math.PI/180)));
    } else if(bonds.length===2) {
      const bisector=norm(add(bonds[0],bonds[1]));
      dir=planar?bisector:add(norm(cross(bonds[0],bonds[1])),scale(bisector,Math.SQRT1_2));
    } else dir=bonds.reduce(add,[0,0,0]);
    dir=norm(dir);
    const [x,y,z]=add(center,scale(dir,length));
    out.push({x,y,z,el:"H"});bonds.push(scale(dir,-1));
  }
  return out;
}
