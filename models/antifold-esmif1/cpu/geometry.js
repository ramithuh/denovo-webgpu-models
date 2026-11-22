// Geometry math mirrors upstream FP32 intermediates, including its two normalizers.
const f=Math.fround,add=(a,b)=>f(a+b),sub=(a,b)=>f(a-b),mul=(a,b)=>f(a*b),div=(a,b)=>f(a/b);
const dot=(a,b)=>add(add(mul(a[0],b[0]),mul(a[1],b[1])),mul(a[2],b[2]));
const minus=(a,b)=>a.map((v,i)=>sub(v,b[i]));
const cross=(a,b)=>[sub(mul(a[1],b[2]),mul(a[2],b[1])),sub(mul(a[2],b[0]),mul(a[0],b[2])),sub(mul(a[0],b[1]),mul(a[1],b[0]))];
const norm=(a,kind=0)=>{const d=kind?Math.max(f(Math.sqrt(dot(a,a))),1e-12):f(Math.sqrt(add(dot(a,a),1e-8)));return a.map(v=>div(v,d));};
function dihedral(coords,N,kind){
  const u=[];for(let j=0;j<3*N-1;j++)u.push(norm(minus(coords.slice((j+1)*3,(j+2)*3),coords.slice(j*3,(j+1)*3)),kind));
  const angles=new Float32Array(N*3);
  for(let j=0;j<u.length-2;j++){
    const n2=norm(cross(u[j],u[j+1]),kind),n1=norm(cross(u[j+1],u[j+2]),kind);
    const co=Math.max(f(-1+1e-7),Math.min(f(1-1e-7),dot(n2,n1)));
    angles[j+1]=mul(Math.sign(dot(u[j],n1)),f(Math.acos(co)));
  }
  const out=new Float32Array(N*6);for(let r=0;r<N;r++)for(let j=0;j<3;j++){out[r*6+j]=Math.cos(angles[r*3+j]);out[r*6+3+j]=Math.sin(angles[r*3+j]);}return out;
}
export function geometry(input){
  const {N,coordMask,padding,confidence}=input,K=Math.min(30,N);
  const coords=Array.from(input.coords,v=>Number.isFinite(v)?v:0),ca=Array.from({length:N},(_,i)=>coords.slice(i*9+3,i*9+6));
  const ds=dihedral(coords,N,0),dihedrals=dihedral(coords,N,1),nodeScalar=new Float32Array(N*7),nodeVector=new Float32Array(N*9),frames=new Float32Array(N*9),confidenceRBF=new Float32Array(N*16);
  for(let r=0;r<N;r++){
    nodeScalar.set(ds.subarray(r*6,(r+1)*6),r*7);nodeScalar[r*7+6]=coordMask[r];
    const n=minus(coords.slice(r*9,r*9+3),ca[r]),c=minus(coords.slice(r*9+6,r*9+9),ca[r]);
    const cn=norm(c),nn=norm(n),bis=norm(cn.map((v,j)=>add(v,nn[j]))),perp=norm(cross(cn,nn));
    const side=bis.map((v,j)=>sub(mul(-v,Math.sqrt(1/3)),mul(perp[j],Math.sqrt(2/3))));
    if(r<N-1)nodeVector.set(norm(minus(ca[r+1],ca[r])),r*9);
    if(r>0)nodeVector.set(norm(minus(ca[r-1],ca[r])),r*9+3);
    nodeVector.set(side,r*9+6);
    const e1=cn,e2=norm(n.map((v,j)=>sub(v,mul(e1[j],dot(e1,n))))),e3=cross(e1,e2);
    frames.set([...e1,...e2,...e3],r*9);
    for(let j=0;j<16;j++){const z=div(sub(confidence[r],f(j/15)),1/16);confidenceRBF[r*16+j]=Math.exp(-mul(z,z));}
  }
  const edges=[],scalars=[],vectors=[];
  for(let src=0;src<N;src++){
    const neighbors=Array.from({length:N},(_,dst)=>{
      const valid=coordMask[src]&&coordMask[dst],res=!padding[src]&&!padding[dst];
      const d=valid?f(Math.sqrt(add(dot(minus(ca[src],ca[dst]),minus(ca[src],ca[dst])),1e-8))):0;
      return {dst,d,sort:add(add(d,valid?0:add(1e8,mul(Math.abs(src-dst),1e6))),res?0:1e10),valid};
    }).sort((a,b)=>a.sort-b.sort||a.dst-b.dst).slice(0,K);
    for(const {dst,d,valid} of neighbors){
      if(!valid||padding[src]||padding[dst])continue;
      edges.push(src,dst);
      for(let j=0;j<16;j++){const z=div(sub(d,f(j*20/15)),1.25);scalars.push(f(Math.exp(-mul(z,z))));}
      const angles=Array.from({length:8},(_,j)=>mul(src-dst,f(Math.exp(mul(j*2,-Math.log(10000)/16)))));
      scalars.push(...angles.map(v=>f(Math.cos(v))),...angles.map(v=>f(Math.sin(v))),0,0);
      vectors.push(...norm(minus(ca[src],ca[dst])));
    }
  }
  return {nodeScalar,nodeVector,dihedrals,frames,confidenceRBF,edgeIndex:Int32Array.from(edges),edgeScalar:Float32Array.from(scalars),edgeVector:Float32Array.from(vectors),E:edges.length/2};
}
