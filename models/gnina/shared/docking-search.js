import {coordinates,projectGradient} from "./torsions.js";
import {sampleVinaGrid,yieldTask} from "./vina-grid.js";

export function seededRandom(seed) {
  let state=seed>>>0;
  return ()=>{
    state=(state+0x6d2b79f5)>>>0;
    let t=Math.imul(state^(state>>>15),state|1);
    t^=t+Math.imul(t^(t>>>7),t|61);
    return ((t^(t>>>14))>>>0)/4294967296;
  };
}
const tau=2*Math.PI;
function randomRotation(q,rng) {
  // Unit quaternion sampled from Haar measure on SO(3).
  const u=rng(),a=tau*rng(),b=tau*rng(),s=Math.sqrt(1-u),t=Math.sqrt(u);
  const w=t*Math.cos(b),h=2*Math.acos(w)/Math.sqrt(1-w*w||1);
  q[3]=s*Math.sin(a)*h;q[4]=s*Math.cos(a)*h;q[5]=t*Math.sin(b)*h;
}
// Compose a space-fixed angular perturbation with the current root rotation.
function turn(q,x,y,z) {
  const a=Math.hypot(x,y,z),b=Math.hypot(q[3],q[4],q[5]);
  const u=a?Math.sin(a/2)/a:.5,v=b?Math.sin(b/2)/b:.5;
  const c=Math.cos(a/2),d=Math.cos(b/2),i=v*q[3],j=v*q[4],k=v*q[5];
  x*=u;y*=u;z*=u;
  const w=c*d-x*i-y*j-z*k;
  const h=2*Math.acos(Math.max(-1,Math.min(1,w)))/Math.sqrt(Math.max(1e-30,1-w*w));
  q[3]=h*(x*d+c*i+y*k-z*j);q[4]=h*(y*d+c*j+z*i-x*k);q[5]=h*(z*d+c*k+x*j-y*i);
}

export function randomConformation(tree,box,rng) {
  const q=new Float64Array(6+tree.rotors.length);
  for(let d=0;d<3;d++) q[d]=box.center[d]+(rng()-.5)*box.size[d];
  randomRotation(q,rng);
  for(let d=6;d<q.length;d++) q[d]=(rng()-.5)*tau;
  return q;
}
export function intramolecular(tree,p,g,cap=Infinity) {
  let energy=0;
  const {pairs,pairTypes,tables}=tree;
  for(let k=0;k<pairTypes.length;k++) {
    const i=pairs[2*k],j=pairs[2*k+1],x=p[i]-p[j],y=p[i+1]-p[j+1],z=p[i+2]-p[j+2];
    const sq=x*x+y*y+z*z;
    if(sq>=64) continue;
    const s=256*sq,bin=s|0,table=tables[pairTypes[k]],slope=table[bin+1]-table[bin];
    const e=table[bin]+(s-bin)*slope,scale=e>0?1/(1+e/cap):1,w=512*slope*scale*scale;
    energy+=e*scale;
    g[i]+=w*x;g[i+1]+=w*y;g[i+2]+=w*z;g[j]-=w*x;g[j+1]-=w*y;g[j+2]-=w*z;
  }
  return energy;
}
export function dockingObjective(tree,grid,intermolecular=(p,g)=>sampleVinaGrid(grid,p,g),cap=Infinity) {
  const p=new Float64Array(3*tree.ligand.atoms.length),force=new Float64Array(p.length);
  return (q,g)=>{
    coordinates(tree,q,p,false);
    const energy=intermolecular(p,force)+intramolecular(tree,p,force,cap);
    projectGradient(tree,q,p,force,g);
    return energy;
  };
}

// Inverse BFGS with Armijo backtracking. Buffers are reused for each chain.
export function localMinimizer(evaluate,n) {
  const H=new Float64Array(n*n),g=new Float64Array(n),g1=g.slice(),trial=g.slice();
  const s=g.slice(),y=g.slice(),Hy=g.slice();
  return (q,maxIter)=>{
    H.fill(0);
    for(let i=0;i<n;i++) H[i*n+i]=1;
    let energy=evaluate(q,g);
    for(let iter=0;iter<maxIter;iter++) {
      let slope=0,maxStep=0;
      for(let i=0;i<n;i++) {
        let v=0;
        for(let j=0;j<n;j++) v-=H[i*n+j]*g[j];
        s[i]=v;slope+=v*g[i];maxStep=Math.max(maxStep,Math.abs(v));
      }
      if(slope>=-1e-8) break;
      let alpha=Math.min(1,2/maxStep),next,accepted=false;
      for(let line=0;line<12;line++,alpha*=.5) {
        for(let i=0;i<n;i++) trial[i]=q[i]+alpha*s[i];
        next=evaluate(trial,g1);
        if(next<=energy+1e-4*alpha*slope){accepted=true;break;}
      }
      if(!accepted) break;
      let ys=0,yHy=0;
      for(let i=0;i<n;i++){s[i]*=alpha;y[i]=g1[i]-g[i];ys+=s[i]*y[i];}
      q.set(trial);g.set(g1);
      const improvement=energy-next;
      energy=next;
      if(improvement<1e-6) break;
      if(ys<=1e-10) continue;
      if(iter===0) {
        let yy=0;
        for(let i=0;i<n;i++) yy+=y[i]*y[i];
        for(let i=0;i<n;i++) H[i*n+i]=ys/yy;
      }
      for(let i=0;i<n;i++) {
        let v=0;
        for(let j=0;j<n;j++) v+=H[i*n+j]*y[j];
        Hy[i]=v;yHy+=y[i]*v;
      }
      const scale=(1+yHy/ys)/ys;
      for(let i=0;i<n;i++) for(let j=0;j<=i;j++) {
        const h=H[i*n+j]+scale*s[i]*s[j]-(Hy[i]*s[j]+s[i]*Hy[j])/ys;
        H[i*n+j]=h;H[j*n+i]=h;
      }
    }
    return energy;
  };
}

export function rmsd(p,r) {
  let sum=0;
  for(let i=0;i<p.length;i++) sum+=(p[i]-r[i])**2;
  return Math.sqrt(sum/(p.length/3));
}

// Remove every neighbor of a newly inserted minimum to keep all modes separated.
function insertCandidate(bank,energy,q,coords,limit,minRmsd) {
  if(bank.length===limit && energy>=bank[limit-1].energy) return;
  for(const v of bank) if(v.energy<=energy && rmsd(coords,v.coords)<minRmsd) return;
  for(let i=bank.length-1;i>=0;i--) if(rmsd(coords,bank[i].coords)<minRmsd) bank.splice(i,1);
  bank.push({energy,q:q.slice(),coords:coords.slice()});bank.sort((a,b)=>a.energy-b.energy);
  if(bank.length>limit) bank.pop();
}

export async function searchChain(tree,grid,options,id,check) {
  const rng=seededRandom(options.seed+Math.imul(id,0x9e3779b9)),bank=[];
  const current=randomConformation(tree,grid.box,rng),trial=current.slice(),k=tree.rotors.length;
  const p=new Float64Array(3*tree.ligand.atoms.length);
  const force=p.slice();
  let gyration=1;
  // A bounded positive potential lets a trial escape initial steric clashes.
  // Relax promising survivors against the full potential before archiving.
  const hunt=localMinimizer(dockingObjective(tree,grid,(p,g)=>sampleVinaGrid(grid,p,g,10),10),trial.length);
  const minimize=localMinimizer(dockingObjective(tree,grid),trial.length);
  const chainLimit=Math.max(50,options.limit);
  let energy=Infinity,lastYield=performance.now();
  for(let step=0;step<options.steps;step++) {
    trial.set(current);
    if(step) {
      const entity=Math.floor(rng()*(k+2));
      if(entity<2) {
        let x,y,z;
        do {x=2*rng()-1;y=2*rng()-1;z=2*rng()-1;} while(x*x+y*y+z*z>1);
        if(entity===0){trial[0]+=2*x;trial[1]+=2*y;trial[2]+=2*z;}
        else turn(trial,2*x/gyration,2*y/gyration,2*z/gyration);
      } else trial[4+entity]=(rng()-.5)*tau;
    }
    const hunted=hunt(trial,Math.ceil((25+tree.ligand.atoms.length)/3));
    coordinates(tree,trial,p,false);
    const next=sampleVinaGrid(grid,p,force);
    if(next<energy || rng()<Math.exp((energy-next)/1.2)) {
      let minimized=hunted;
      if(bank.length<chainLimit || hunted<bank[bank.length-1].energy){
        minimized=minimize(trial,25);
        coordinates(tree,trial,p,false);
      }
      current.set(trial);energy=sampleVinaGrid(grid,p,force);
      let radius2=0;
      for(let i=0;i<p.length;i++) radius2+=(p[i]-current[i%3])**2;
      gyration=Math.max(1,Math.sqrt(radius2/(p.length/3)));
      // Equivalent rotation vectors outside the principal ball become ill-conditioned.
      const t=Math.hypot(current[3],current[4],current[5]);
      if(t>Math.PI) for(let i=3;i<6;i++) current[i]*=(t-tau*Math.round(t/tau))/t;
      if(bank.length<chainLimit || minimized<bank[bank.length-1].energy) {
        insertCandidate(bank,minimized,current,p,chainLimit,options.minRmsd);
      }
    }
    if(check && performance.now()-lastYield>45){await yieldTask();check();lastYield=performance.now();}
  }
  return bank;
}
