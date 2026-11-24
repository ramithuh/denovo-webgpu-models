// Jacobi 3x3 SVD and proper-rotation Kabsch. All matrices use row-major storage.
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const determinant=m=>dot(m.slice(0,3),cross(m.slice(3,6),m.slice(6,9)));
export function svd3(matrix){
  if(matrix.length!==9||!Array.from(matrix).every(Number.isFinite))throw new Error('Expected finite 3x3 matrix');
  const a=Array.from(matrix),v=[1,0,0,0,1,0,0,0,1];
  // One-sided Jacobi orthogonalizes columns without squaring the condition number.
  for(let sweep=0;sweep<32;sweep++){
    let changed=false;
    for(const [p,q]of [[0,1],[0,2],[1,2]]){
      let alpha=0,beta=0,gamma=0;for(let r=0;r<3;r++){alpha+=a[r*3+p]**2;beta+=a[r*3+q]**2;gamma+=a[r*3+p]*a[r*3+q];}
      if(Math.abs(gamma)<=1e-15*Math.sqrt(alpha*beta))continue;
      changed=true;const z=(beta-alpha)/(2*gamma),t=(z>=0?1:-1)/(Math.abs(z)+Math.hypot(1,z)),c=1/Math.hypot(1,t),s=c*t;
      for(const m of [a,v])for(let r=0;r<3;r++){const x=m[r*3+p],y=m[r*3+q];m[r*3+p]=c*x-s*y;m[r*3+q]=s*x+c*y;}
    }if(!changed)break;
  }
  const order=[0,1,2].sort((p,q)=>Math.hypot(a[q],a[3+q],a[6+q])-Math.hypot(a[p],a[3+p],a[6+p]));
  const singular=order.map(c=>Math.hypot(a[c],a[3+c],a[6+c])),cols=[];
  for(let j=0;j<3;j++){
    let col=singular[j]>Math.max(1e-14,singular[0]*1e-12)?[a[order[j]],a[3+order[j]],a[6+order[j]]].map(x=>x/singular[j]):null;
    if(!col){const axes=[[1,0,0],[0,1,0],[0,0,1]];col=axes.sort((x,y)=>cols.reduce((s,u)=>s+dot(x,u)**2,0)-cols.reduce((s,u)=>s+dot(y,u)**2,0))[0].slice();}
    for(const prev of cols){const projection=dot(col,prev);col=col.map((x,i)=>x-projection*prev[i]);}const norm=Math.hypot(...col);cols.push(col.map(x=>x/norm));
  }
  return {u:Array.from({length:9},(_,i)=>cols[i%3][Math.floor(i/3)]),s:singular,v:Array.from({length:9},(_,i)=>v[Math.floor(i/3)*3+order[i%3]])};
}
export function rigidAlign(target,moving,{mask,weights}={}){
  if(target.length!==moving.length||target.length%3||!Array.from(target).every(Number.isFinite)||!Array.from(moving).every(Number.isFinite))throw new Error('Invalid alignment coordinates');
  const n=target.length/3,a=[0,0,0],b=[0,0,0],w=Array.from({length:n},(_,i)=>(mask&&!mask[i]?0:weights?.[i]??1));
  if(w.some(v=>!Number.isFinite(v)||v<0))throw new Error('Invalid alignment weights');
  const total=w.reduce((x,y)=>x+y,0);if(total<=0)throw new Error('Alignment needs at least one selected atom');
  for(let i=0;i<n;i++)for(let c=0;c<3;c++){a[c]+=w[i]*target[i*3+c]/total;b[c]+=w[i]*moving[i*3+c]/total;}
  const cov=Array(9).fill(0);for(let i=0;i<n;i++)for(let r=0;r<3;r++)for(let c=0;c<3;c++)cov[r*3+c]+=w[i]*(moving[i*3+r]-b[r])*(target[i*3+c]-a[c]);
  const {u,v}=svd3(cov),sign=determinant(u)*determinant(v)<0?-1:1,rotation=Array(9).fill(0);
  for(let r=0;r<3;r++)for(let c=0;c<3;c++)for(let k=0;k<3;k++)rotation[r*3+c]+=u[r*3+k]*v[c*3+k]*(k===2?sign:1);
  const coordinates=Float32Array.from(moving,(_,i)=>a[i%3]+[0,1,2].reduce((s,r)=>s+(moving[Math.floor(i/3)*3+r]-b[r])*rotation[r*3+i%3],0));
  return {coordinates,rotation,targetCenter:a,movingCenter:b};
}
export function rigidAugment(x,base,fixed,{angles,translation,centerOption='all',centeringAffectsMotif=true,reinsertMotif=true,sTrans=1}={}){
  if(!['all','motif','diffuse'].includes(centerOption))throw new Error('Invalid centering option');
  if(angles?.length!==3||translation?.length!==3||![...angles,...translation,sTrans].every(Number.isFinite))throw new Error('Invalid rigid augmentation draw');
  const out=x.slice(),hasFixed=fixed.some(Boolean),n=x.length/3;
  if(hasFixed&&reinsertMotif){const aligned=rigidAlign(out,base,{mask:fixed}).coordinates;for(let i=0;i<n;i++)if(fixed[i])out.set(aligned.subarray(i*3,i*3+3),i*3);}
  const selected=Array.from({length:n},(_,i)=>i).filter(i=>!hasFixed||centerOption==='all'||Boolean(fixed[i])===(centerOption==='motif'));
  if(!selected.length)throw new Error('Cannot center an empty atom selection');
  const center=[0,0,0];for(const i of selected)for(let c=0;c<3;c++)center[c]+=out[i*3+c]/selected.length;
  const [ax,ay,az]=angles,[cx,cy,cz]=angles.map(Math.cos),[sx,sy,sz]=angles.map(Math.sin);
  const rotation=[cz*cy,cz*sy*sx-sz*cx,cz*sy*cx+sz*sx,sz*cy,sz*sy*sx+cz*cx,sz*sy*cx-cz*sx,-sy,cy*sx,cy*cx];
  for(let i=0;i<n;i++){const v=[0,1,2].map(c=>out[i*3+c]-(centeringAffectsMotif||!fixed[i]?center[c]:0));for(let r=0;r<3;r++)out[i*3+r]=dot(rotation.slice(r*3,r*3+3),v)+translation[r]*sTrans;}
  return {coordinates:out,rotation};
}
