// Symmetric 4x4 eigensolver: lower Householder reduction, implicit QL/QR.
// Matches the LAPACK algorithm family used by torch.linalg.eigh (UPLO=L),
// with deterministic quaternion sign canonicalization (first |q_k| > 1e-7 positive)
// matching genie3-webgpu to eliminate LAPACK eigenvector sign ambiguity.
import { portable } from "./portable.js";
export const eigenBody = `
var a:array<f32,16>;var v:array<f32,16>;var d:array<f32,4>;var e:array<f32,4>;
var h:array<f32,4>;var w:array<f32,4>;var cs:array<f32,4>;var sn:array<f32,4>;
for(var j=0u;j<16u;j++){a[j]=a0[base+j];v[j]=0.0;}for(var j=0u;j<4u;j++){v[j*4u+j]=1.0;}
for(var k=0u;k<3u;k++){
  let n=3u-k;let alpha=a[(k+1u)*4u+k];var norm=0.0;
  for(var j=k+2u;j<4u;j++){norm=norm+a[j*4u+k]*a[j*4u+k];}
  var tau=0.0;var beta=alpha;
  for(var j=0u;j<4u;j++){h[j]=0.0;}
  h[k+1u]=1.0;
  if(norm>0.0){beta=sqrt(alpha*alpha+norm);if(alpha>=0.0){beta=-beta;}
    tau=rfdiv(beta-alpha,beta);
    for(var j=k+2u;j<4u;j++){h[j]=a[j*4u+k]*rfdiv(1.0,alpha-beta);}
  }
  if(tau!=0.0){
    for(var j=k+1u;j<4u;j++){var z=0.0;for(var r=k+1u;r<4u;r++){z=fma(a[j*4u+r],h[r],z);}w[j]=tau*z;}
    var z=0.0;for(var j=k+1u;j<4u;j++){z=fma(w[j],h[j],z);}let fac=-0.5*tau*z;
    for(var j=k+1u;j<4u;j++){w[j]=fma(fac,h[j],w[j]);}
    for(var j=k+1u;j<4u;j++){for(var r=j;r<4u;r++){let z=(a[r*4u+j]-h[r]*w[j])-w[r]*h[j];a[r*4u+j]=z;a[j*4u+r]=z;}}
    for(var j=0u;j<4u;j++){var z=0.0;for(var r=k+1u;r<4u;r++){z=fma(v[j*4u+r],h[r],z);}z=tau*z;for(var r=k+1u;r<4u;r++){v[j*4u+r]=v[j*4u+r]-z*h[r];}}
  }
  d[k]=a[k*4u+k];e[k]=beta;
}d[3]=a[15];e[3]=0.0;
var start=0i;var iterations=0u;
loop {
  if(start>=4i){break;}
  var end=start;
  for(var j=start;j<3i;j++){
    let threshold=5.960464477539063e-8*sqrt(abs(d[j]))*sqrt(abs(d[j+1i]));
    if(abs(e[j])<=threshold){e[j]=0.0;break;}end=j+1i;
  }
  var l=start;var last=end;start=end+1i;
  if(abs(d[last])<abs(d[l])){let tmp=l;l=last;last=tmp;}
  var direction=1i;if(last<l){direction=-1i;}
  loop {
    if((direction==1i&&l>last)||(direction==-1i&&l<last)){break;}
    var m=l;
    loop {
      if(m==last){break;}
      var ix=m;if(direction==-1i){ix=m-1i;}
      if(e[ix]*e[ix]<=3.552713678800501e-15*abs(d[m])*abs(d[m+direction])+1.1754943508222875e-38){e[ix]=0.0;break;}
      m=m+direction;
    }
    if(m==l){l=l+direction;continue;}
    if(abs(m-l)==1i){
      let ix=min(l,m);let aa=d[ix];let bb=e[ix];let cc=d[ix+1i];
      let sm=aa+cc;let df=aa-cc;let tb=2.0*bb;let adf=abs(df);let ab=abs(tb);
      var rt=0.0;if(adf>ab){rt=adf*sqrt(1.0+(ab/adf)*(ab/adf));}else if(adf<ab){rt=ab*sqrt(1.0+(adf/ab)*(adf/ab));}else{rt=ab*sqrt(2.0);}
      var rt1=0.5*(sm+rt);var rt2=0.0;var sign1=1.0;
      if(sm<0.0){rt1=0.5*(sm-rt);sign1=-1.0;}
      if(sm==0.0){rt2=-0.5*rt;}else{var acmx=aa;var acmn=cc;if(abs(cc)>abs(aa)){acmx=cc;acmn=aa;}rt2=(acmx/rt1)*acmn-(bb/rt1)*bb;}
      var ctmp=df+rt;var sign2=1.0;if(df<0.0){ctmp=df-rt;sign2=-1.0;}
      var c=1.0;var s=0.0;
      if(abs(ctmp)>ab){let ct=-tb/ctmp;s=1.0/sqrt(1.0+ct*ct);c=ct*s;}else if(ab!=0.0){let tn=-ctmp/tb;c=1.0/sqrt(1.0+tn*tn);s=tn*c;}
      if(sign1==sign2){let temp=c;c=-s;s=temp;}
      for(var j=0i;j<4i;j++){let x=v[j*4i+ix];let y=v[j*4i+ix+1i];v[j*4i+ix]=c*x+s*y;v[j*4i+ix+1i]=c*y-s*x;}
      d[ix]=rt1;d[ix+1i]=rt2;e[ix]=0.0;l=l+2i*direction;continue;
    }
    if(iterations>=120u){break;}iterations++;
    var ix=l;if(direction==-1i){ix=l-1i;}
    var p=d[l];var g=(d[l+direction]-p)/(2.0*e[ix]);var r=sqrt(g*g+1.0);if(g<0.0){r=-r;}
    g=d[m]-p+e[ix]/(g+r);var s=1.0;var c=1.0;p=0.0;
    var j=m-direction;
    loop {
      if((direction==1i&&j<l)||(direction==-1i&&j>l)){break;}
      var k=j;if(direction==-1i){k=j-1i;}
      let ff=s*e[k];let b=c*e[k];
      if(ff==0.0){c=1.0;s=0.0;r=g;}else if(g==0.0){c=0.0;s=1.0;if(ff<0.0){s=-1.0;}r=abs(ff);}else{r=sqrt(g*g+ff*ff);if(g<0.0){r=-r;}c=g/r;s=ff/r;}
      if(j!=m-direction){e[k+direction]=r;}
      var di=j+1i;var other=j;if(direction==-1i){di=j-1i;other=j;}
      g=d[di]-p;r=(d[other]-g)*s+2.0*c*b;p=s*r;d[di]=g+p;g=c*r-b;
      cs[k]=c;sn[k]=-s;if(direction==-1i){sn[k]=s;}
      j=j-direction;
    }
    j=m-direction;
    loop {
      if((direction==1i&&j<l)||(direction==-1i&&j>l)){break;}
      var k=j;if(direction==-1i){k=j-1i;}
      for(var row=0i;row<4i;row++){let x=v[row*4i+k];let y=v[row*4i+k+1i];v[row*4i+k]=cs[k]*x+sn[k]*y;v[row*4i+k+1i]=cs[k]*y-sn[k]*x;}
      j=j-direction;
    }
    d[l]=d[l]-p;e[ix]=g;
  }
}
var best=0u;for(var j=1u;j<4u;j++){if(d[j]>d[best]){best=j;}}
let q0=v[best];let q1=v[4u+best];let q2=v[8u+best];let q3=v[12u+best];
var signFlip=1.0;
if(abs(q0)>1e-7){if(q0<0.0){signFlip=-1.0;}}
else if(abs(q1)>1e-7){if(q1<0.0){signFlip=-1.0;}}
else if(abs(q2)>1e-7){if(q2<0.0){signFlip=-1.0;}}
else if(abs(q3)>1e-7){if(q3<0.0){signFlip=-1.0;}}
o[i]=v[component*4u+best]*signFlip;
`;
export function eigQuaternion(o, k) {
  const L = k.shape[0];
  return portable(
    o,
    [L, 4],
    [k],
    `let base=u32(floor(f32(i)/4.0))*16u;let component=i%4u;` + eigenBody,
  );
}
export function orientations(o, rots, mask) {
  const L = rots.shape[0];
  // This order deliberately preserves upstream R_j @ R_i (no transpose).
  const r = o.einsum("jab,ibc->ijac", rots, rots);
  const k = portable(
    o,
    [L * L, 16],
    [r],
    `
    let base=u32(floor(f32(i)/16.0))*9u;let c=i%16u;
    let xx=a0[base];let xy=a0[base+1u];let xz=a0[base+2u];
    let yx=a0[base+3u];let yy=a0[base+4u];let yz=a0[base+5u];
    let zx=a0[base+6u];let zy=a0[base+7u];let zz=a0[base+8u];var z=0.0;
    if(c==0u){z=(xx+yy)+zz;}else if(c==5u){z=(xx-yy)-zz;}else if(c==10u){z=(yy-xx)-zz;}else if(c==15u){z=(zz-xx)-yy;}
    else if(c==1u||c==4u){z=zy-yz;}else if(c==2u||c==8u){z=xz-zx;}else if(c==3u||c==12u){z=yx-xy;}
    else if(c==6u||c==9u){z=xy+yx;}else if(c==7u||c==13u){z=xz+zx;}else{z=yz+zy;}
    o[i]=z*0.3333333333333333;`,
  );
  return o.mul(
    o.reshape(eigQuaternion(o, k), [L, L, 4]),
    o.reshape(o.einsum("i,j->ij", mask, mask), [L, L, 1]),
  );
}
