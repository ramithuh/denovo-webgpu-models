import {pairTable,typeKey} from "./vina-grid.js";

// Cartesian coordinates are packed heavy atoms first, then center-only hydrogens.
// All conformations use an absolute root position and a rotation vector (radians).
const xyz=a=>[a.x,a.y,a.z];
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=a=>{const n=Math.sqrt(dot(a,a));return a.map(x=>x/n);};
const delta=(coords,i,j)=>[0,1,2].map(d=>coords[3*i+d]-coords[3*j+d]);

// Rodrigues' formula, also used to fix each reference dihedral to zero.
function twist(p,bond,angle,count) {
  const a=bond.a*3, b=bond.b*3, x=p[a],y=p[a+1],z=p[a+2];
  const u=(p[b]-x)*bond.invLength,v=(p[b+1]-y)*bond.invLength,w=(p[b+2]-z)*bond.invLength;
  const c=Math.cos(angle),s=Math.sin(angle),t=1-c;
  for(let k=0;k<count;k++) {
    const i=bond.moving[k]*3, dx=p[i]-x,dy=p[i+1]-y,dz=p[i+2]-z, h=t*(u*dx+v*dy+w*dz);
    p[i]=x+c*dx+s*(v*dz-w*dy)+u*h;
    p[i+1]=y+c*dy+s*(w*dx-u*dz)+v*h;
    p[i+2]=z+c*dz+s*(u*dy-v*dx)+w*h;
  }
}

export function buildTorsionTree(ligand) {
  const {atoms,bonds,rotorBonds,centerAtoms,centerParents}=ligand;
  if(!bonds || !rotorBonds || !centerParents) throw Error("Docking requires ligand connectivity from typeMolecule");
  const n=atoms.length, indices=new Map(atoms.map((a,i)=>[a.index,i])), edges=atoms.map(()=>[]);
  const rotatable=new Set(rotorBonds);
  bonds.forEach(([ai,bi],index)=>{
    const a=indices.get(ai),b=indices.get(bi);
    if(a===undefined||b===undefined) return;
    const rotor=rotatable.has(index);
    edges[a].push({to:b,rotor});edges[b].push({to:a,rotor});
  });
  const fragment=new Int32Array(n).fill(-1), groups=[];
  function collect(i,id) {
    fragment[i]=id;groups[id].push(i);
    for(const e of edges[i]) if(!e.rotor && fragment[e.to]<0) collect(e.to,id);
  }
  for(let i=0;i<n;i++) if(fragment[i]<0) {groups.push([]);collect(i,groups.length-1);}
  const links=groups.map(()=>[]);
  for(let a=0;a<n;a++) for(const {to:b,rotor} of edges[a]) if(rotor) links[fragment[a]].push({a,b,to:fragment[b]});
  function weight(id,parent) {
    return links[id].reduce((s,e)=>s+(e.to===parent?0:weight(e.to,id)),groups[id].length);
  }
  // Weighted tree centroid minimizes the size of the largest mobile branch.
  let root=0,best=Infinity;
  for(let i=0;i<groups.length;i++) {
    const largest=Math.max(0,...links[i].map(e=>weight(e.to,i)));
    if(largest<best || (largest===best && groups[i].length>groups[root].length)) {best=largest;root=i;}
  }
  if(weight(root,-1)!==n) throw Error("Dock one connected ligand at a time");
  const points=atoms.map(xyz),owners=Array.from(atoms,(_,i)=>i);
  const centerIndices=centerAtoms.map((a,i)=>{
    const owner=indices.get(centerParents[i]);
    if(owner===undefined) throw Error("Missing center atom parent");
    if(a.el!=="H") return owner;
    owners.push(owner);points.push(xyz(a));return points.length-1;
  });
  const origin=[0,1,2].map(d=>groups[root].reduce((s,i)=>s+points[i][d],0)/groups[root].length);
  // Root frame uses rigid atoms and neighboring bond endpoints. This removes
  // any input rigid-body transform even when the root is a single atom.
  const rootHydrogens=owners.flatMap((owner,i)=>i>=n&&fragment[owner]===root?[i]:[]);
  const anchors=[...groups[root],...links[root].map(e=>e.b),...rootHydrogens].map(i=>points[i].map((v,d)=>v-origin[d]));
  const x=unit(anchors.find(v=>dot(v,v)>1e-12)??[1,0,0]);
  const z=unit(cross(x,anchors.find(v=>dot(cross(x,v),cross(x,v))>1e-12)??(Math.abs(x[0])<.8?[1,0,0]:[0,1,0])));
  const y=cross(z,x);
  const base=Float64Array.from(points.flatMap(p=>{
    const v=p.map((s,d)=>s-origin[d]);return [dot(x,v),dot(y,v),dot(z,v)];
  }));
  const rotors=[];
  function visit(id,parent) {
    const members=groups[id].slice();
    for(const {a,b,to} of links[id]) if(to!==parent) {
      const bond={a,b,invLength:1/Math.hypot(...delta(base,b,a))};
      if(!Number.isFinite(bond.invLength)) throw Error("Zero-length rotatable bond");
      rotors.push(bond);
      const branch=visit(to,id),owned=new Set(branch);
      bond.heavyCount=branch.length;
      bond.moving=Int32Array.from(branch.concat(owners.flatMap((owner,i)=>i>=n&&owned.has(owner)?[i]:[])));
      members.push(...branch);
    }
    return members;
  }
  visit(root,-1);
  for(const bond of rotors) {
    const {a,b}=bond,u=unit(delta(base,b,a));
    const ref=(i,j)=>delta(base,edges[i].find(e=>e.to!==j).to,i);
    const left=cross(u,ref(a,b)),right=cross(u,ref(b,a));
    twist(base,bond,-Math.atan2(dot(u,cross(left,right)),dot(left,right)),bond.moving.length);
  }
  const pairs=[],pairTypes=[],tables=[],typeIndices=new Map();
  for(let i=0;i<n;i++) {
    const near=new Set([i]);let shell=[i];
    for(let d=0;d<3;d++){
      const next=[];
      for(const a of shell) for(const {to} of edges[a]) if(!near.has(to)){near.add(to);next.push(to);}
      shell=next;
    }
    for(let j=i+1;j<n;j++) if(fragment[i]!==fragment[j]&&!near.has(j)) {
      const key=[typeKey(atoms[i]),typeKey(atoms[j])].sort().join("|");
      if(!typeIndices.has(key)){typeIndices.set(key,tables.length);tables.push(pairTable(atoms[i],atoms[j]));}
      pairs.push(3*i,3*j);pairTypes.push(typeIndices.get(key));
    }
  }
  return {ligand,base,root:groups[root],rotors,pairs:Int32Array.from(pairs),pairTypes:Int32Array.from(pairTypes),tables,centerIndices};
}

export function coordinates(tree,q,out=new Float64Array(tree.base.length),includeHydrogens=true) {
  const length=includeHydrogens?tree.base.length:tree.ligand.atoms.length*3;
  const x=q[3],y=q[4],z=q[5],t2=x*x+y*y+z*z,t=Math.sqrt(t2);
  const a=t2<1e-8?1-t2/6:Math.sin(t)/t, b=t2<1e-8?.5-t2/24:(1-Math.cos(t))/t2;
  const base=tree.base;
  for(let i=0;i<length;i+=3) {
    const dx=base[i],dy=base[i+1],dz=base[i+2];
    const ux=y*dz-z*dy,uy=z*dx-x*dz,uz=x*dy-y*dx;
    out[i]=dx+a*ux+b*(y*uz-z*uy)+q[0];
    out[i+1]=dy+a*uy+b*(z*ux-x*uz)+q[1];
    out[i+2]=dz+a*uz+b*(x*uy-y*ux)+q[2];
  }
  for(let k=0;k<tree.rotors.length;k++){
    const r=tree.rotors[k];twist(out,r,q[k+6],includeHydrogens?r.moving.length:r.heavyCount);
  }
  return out;
}

export function dockingPose(tree,q) {
  const p=coordinates(tree,q),lig=tree.ligand;
  const move=(a,i)=>({...a,x:p[i*3],y:p[i*3+1],z:p[i*3+2]});
  return {...lig,atoms:lig.atoms.map(move),centerAtoms:lig.centerAtoms.map((a,i)=>move(a,tree.centerIndices[i]))};
}

// Chain rule: net force, root torque (SO(3) left Jacobian transpose),
// then the torque of each downstream subtree about its current bond axis.
export function projectGradient(tree,q,p,force,g) {
  let fx=0,fy=0,fz=0,tx=0,ty=0,tz=0;
  for(let i=0;i<force.length;i+=3) {
    const x=p[i]-q[0],y=p[i+1]-q[1],z=p[i+2]-q[2],u=force[i],v=force[i+1],w=force[i+2];
    fx+=u;fy+=v;fz+=w;tx+=y*w-z*v;ty+=z*u-x*w;tz+=x*v-y*u;
  }
  g[0]=fx;g[1]=fy;g[2]=fz;
  const x=q[3],y=q[4],z=q[5],t2=x*x+y*y+z*z,t=Math.sqrt(t2);
  const b=t2<1e-8?.5-t2/24:(1-Math.cos(t))/t2,c=t2<1e-8?1/6-t2/120:(1-Math.sin(t)/t)/t2;
  const u=y*tz-z*ty,v=z*tx-x*tz,w=x*ty-y*tx;
  g[3]=tx-b*u+c*(y*w-z*v);g[4]=ty-b*v+c*(z*u-x*w);g[5]=tz-b*w+c*(x*v-y*u);
  for(let k=0;k<tree.rotors.length;k++) {
    const r=tree.rotors[k],a=3*r.a,b=3*r.b;
    const dx=p[b]-p[a],dy=p[b+1]-p[a+1],dz=p[b+2]-p[a+2];
    let torque=0;
    for(let j=0;j<r.heavyCount;j++) {
      const i=3*r.moving[j],x=p[i]-p[a],y=p[i+1]-p[a+1],z=p[i+2]-p[a+2];
      torque+=force[i]*(dy*z-dz*y)+force[i+1]*(dz*x-dx*z)+force[i+2]*(dx*y-dy*x);
    }
    g[k+6]=torque*r.invLength;
  }
}
