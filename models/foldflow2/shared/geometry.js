export function normalize(o, x, eps = 1e-8) {
  const n = x.shape.at(-1);
  return o.kernel(
    x.shape,
    [x],
    `const r=Math.floor(i/${n})*${n};let s=0;for(let j=0;j<${n};j++)s+=a0[r+j]**2;o[i]=a0[i]/Math.sqrt(Math.max(s,${eps}));`,
    `let r=i/${n}u*${n}u;var s=0.0;for(var j=0u;j<${n}u;j++){s+=a0[r+j]*a0[r+j];}o[i]=a0[i]/sqrt(max(s,${eps}));`,
  );
}
export function rotation(o, q) {
  const L = q.shape[0];
  return o.kernel(
    [L, 3, 3],
    [q],
    `const r=Math.floor(i/9)*4,k=i%9,w=a0[r],x=a0[r+1],y=a0[r+2],z=a0[r+3];o[i]=[w*w+x*x-y*y-z*z,2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),w*w-x*x+y*y-z*z,2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),w*w-x*x-y*y+z*z][k];`,
    `let r=i/9u*4u;let k=i%9u;let w=a0[r];let x=a0[r+1u];let y=a0[r+2u];let z=a0[r+3u];var v:array<f32,9>;v[0]=((w*w+x*x)-y*y)-z*z;v[1]=2.0*(x*y-w*z);v[2]=2.0*(x*z+w*y);v[3]=2.0*(x*y+w*z);v[4]=((w*w-x*x)+y*y)-z*z;v[5]=2.0*(y*z-w*x);v[6]=2.0*(x*z-w*y);v[7]=2.0*(y*z+w*x);v[8]=((w*w-x*x)-y*y)+z*z;o[i]=v[k];`,
  );
}
export function update(o, q, t, u) {
  const L = q.shape[0],
    R = rotation(o, q),
    v = o.slice(u, 1, 3, 6);
  const nq = o.kernel(
    [L, 4],
    [q, u],
    `const r=Math.floor(i/4),k=i%4,w=a0[r*4],x=a0[r*4+1],y=a0[r*4+2],z=a0[r*4+3],a=a1[r*6],b=a1[r*6+1],c=a1[r*6+2];o[i]=a0[i]+[-x*a-y*b-z*c,w*a+y*c-z*b,w*b+z*a-x*c,w*c+x*b-y*a][k];`,
    `let r=i/4u;let k=i%4u;let w=a0[r*4u];let x=a0[r*4u+1u];let y=a0[r*4u+2u];let z=a0[r*4u+3u];let a=a1[r*6u];let b=a1[r*6u+1u];let c=a1[r*6u+2u];var v:array<f32,4>;v[0]=-x*a-y*b-z*c;v[1]=w*a+y*c-z*b;v[2]=w*b+z*a-x*c;v[3]=w*c+x*b-y*a;o[i]=a0[i]+v[k];`,
  );
  return [normalize(o, nq), o.add(t, o.einsum("lij,lj->li", R, v))];
}
export function points(o, x, q, t, H, P) {
  const L = q.shape[0],
    local = o.permute(o.reshape(x, [L, 3, H, P]), [0, 2, 3, 1]);
  return o.add(
    o.einsum("lij,lhpj->lhpi", rotation(o, q), local),
    o.reshape(t, [L, 1, 1, 3]),
  );
}
export function inversePoints(o, x, q, t) {
  return o.einsum(
    "lji,lhpj->lhpi",
    rotation(o, q),
    o.sub(x, o.reshape(t, [q.shape[0], 1, 1, 3])),
  );
}
export function backbone(o, q, t, psi) {
  const L = q.shape[0];
  const local = o.kernel(
    [L, 5, 3],
    [psi],
    `const r=Math.floor(i/15),a=Math.floor(i/3)%5,k=i%3,s=a0[r*2],c=a0[r*2+1];const p=[[-.525,1.363,0],[0,0,0],[1.526,0,0],[-.529,-.774,-1.205],[2.153,-1.062*c,-1.062*s]];o[i]=p[a][k];`,
    `let r=i/15u;let a=i/3u%5u;let k=i%3u;let s=a0[r*2u];let c=a0[r*2u+1u];var p:array<vec3<f32>,5>;p[0]=vec3<f32>(-.525,1.363,0.0);p[1]=vec3<f32>(0.0);p[2]=vec3<f32>(1.526,0.0,0.0);p[3]=vec3<f32>(-.529,-.774,-1.205);p[4]=vec3<f32>(2.153,-1.062*c,-1.062*s);o[i]=p[a][k];`,
  );
  return o.add(
    o.einsum("lij,laj->lai", rotation(o, q), local),
    o.reshape(t, [L, 1, 3]),
  );
}
export function canonical(o, q) {
  return o.kernel(
    q.shape,
    [q],
    `o[i]=a0[Math.floor(i/4)*4]<0?-a0[i]:a0[i];`,
    `o[i]=select(a0[i],-a0[i],a0[i/4u*4u]<0.0);`,
  );
}
export function relativeQuaternion(o, a, b) {
  const L = a.shape[0];
  return o.kernel(
    [L, 4],
    [a, b],
    `const r=Math.floor(i/4)*4,k=i%4,w=a0[r],x=-a0[r+1],y=-a0[r+2],z=-a0[r+3],v=a1[r],a=a1[r+1],b=a1[r+2],c=a1[r+3];o[i]=[w*v-x*a-y*b-z*c,w*a+x*v+y*c-z*b,w*b-x*c+y*v+z*a,w*c+x*b-y*a+z*v][k];`,
    `let r=i/4u*4u;let k=i%4u;let w=a0[r];let x=-a0[r+1u];let y=-a0[r+2u];let z=-a0[r+3u];let v=a1[r];let a=a1[r+1u];let b=a1[r+2u];let c=a1[r+3u];var q:array<f32,4>;q[0]=w*v-x*a-y*b-z*c;q[1]=w*a+x*v+y*c-z*b;q[2]=w*b-x*c+y*v+z*a;q[3]=w*c+x*b-y*a+z*v;o[i]=q[k];`,
  );
}
// Preserve pinned quat_to_rotvec: the 1e-6 inside sin is part of its numerical definition.
export function vectorfields(o, frames, pred, time, rotScale = 10) {
  const L = frames.shape[0],
    q = normalize(o, o.slice(frames, 1, 0, 4)),
    qp = pred.q,
    rel = canonical(o, relativeQuaternion(o, qp, q)),
    R = rotation(o, q),
    s =
      rotScale < 0
        ? 1 / Math.max(-rotScale, Math.min(0.9999, Math.max(0.0001, time)))
        : rotScale;
  const log = o.kernel(
    [L, 3, 3],
    [rel],
    `const r=Math.floor(i/9),k=i%9,w=a0[r*4],x=a0[r*4+1],y=a0[r*4+2],z=a0[r*4+3],n=Math.hypot(x,y,z),angle=2*Math.atan2(n,w),angle2=angle*angle,f=(angle<=1e-3?2+angle2/12+7*angle2*angle2/2880:angle/Math.sin(angle/2+1e-6))/Math.hypot(n,w);o[i]=[0,-z,y,z,0,-x,-y,x,0][k]*f*${s};`,
    `let r=i/9u;let k=i%9u;let w=a0[r*4u];let x=a0[r*4u+1u];let y=a0[r*4u+2u];let z=a0[r*4u+3u];let n=sqrt(x*x+y*y+z*z);let angle=2.0*atan2(n,w);let angle2=angle*angle;var f=2.0+angle2/12.0+7.0*angle2*angle2/2880.0;if(angle>1e-3){f=angle/sin(angle/2.0+1e-6);}f=f/sqrt(n*n+w*w);var a:array<f32,9>;a[0]=0.0;a[1]=-z;a[2]=y;a[3]=z;a[4]=0.0;a[5]=-x;a[6]=-y;a[7]=x;a[8]=0.0;o[i]=a[k]*f*${s};`,
  );
  return {
    rot: o.einsum("lij,ljk->lik", R, log),
    trans: o.div(
      o.sub(o.mul(o.slice(frames, 1, 4, 7), 0.1), o.mul(pred.t, 0.1)),
      time + 1e-10,
    ),
  };
}
export function integrate(o, frames, pred, time, dt, rotScale = 10) {
  const L = frames.shape[0],
    q = normalize(o, o.slice(frames, 1, 0, 4)),
    rel = canonical(o, relativeQuaternion(o, q, pred.q)),
    alpha =
      dt *
      (rotScale < 0
        ? 1 / Math.max(-rotScale, Math.min(0.9999, Math.max(0.0001, time)))
        : rotScale);
  const dq = o.kernel(
    [L, 4],
    [rel],
    `const r=Math.floor(i/4)*4,k=i%4,n=Math.hypot(a0[r+1],a0[r+2],a0[r+3]),theta=2*Math.atan2(n,a0[r]),theta2=theta*theta,scale=theta<=1e-3?2+theta2/12+7*theta2*theta2/2880:theta/Math.sin(theta/2+1e-6),angle=.5*n/Math.hypot(n,a0[r])*scale*${alpha};o[i]=k===0?Math.cos(angle):a0[i]*(n<1e-8?${alpha}:Math.sin(angle)/n);`,
    `let r=i/4u*4u;let k=i%4u;let n=sqrt(a0[r+1u]*a0[r+1u]+a0[r+2u]*a0[r+2u]+a0[r+3u]*a0[r+3u]);let theta=2.0*atan2(n,a0[r]);let theta2=theta*theta;var scale=2.0+theta2/12.0+7.0*theta2*theta2/2880.0;if(theta>1e-3){scale=theta/sin(theta/2.0+1e-6);}let angle=0.5*n/sqrt(n*n+a0[r]*a0[r])*scale*${alpha};var s=${Number(alpha).toExponential()};if(n>1e-8){s=sin(angle)/n;}if(k==0u){o[i]=cos(angle);}else{o[i]=a0[i]*s;}`,
  );
  const conj = o.kernel(
      q.shape,
      [q],
      `o[i]=i%4===0?a0[i]:-a0[i];`,
      `o[i]=select(-a0[i],a0[i],i%4u==0u);`,
    ),
    nq = canonical(o, normalize(o, relativeQuaternion(o, conj, dq)));
  const t = o.slice(frames, 1, 4, 7),
    v = o.div(o.sub(o.mul(t, 0.1), o.mul(pred.t, 0.1)), time + 1e-10),
    nt = o.sub(o.mul(t, 0.1), o.mul(v, dt));
  const centered = o.kernel(
    [L, 3],
    [nt],
    `const c=i%3;let s=0;for(let j=0;j<${L};j++)s+=a0[j*3+c];o[i]=(a0[i]-s/${L})/.1;`,
    `let c=i%3u;var s=0.0;for(var j=0u;j<${L}u;j++){s+=a0[j*3u+c];}o[i]=(a0[i]-s/${L}.0)/0.1;`,
  );
  return o.cat([nq, centered]);
}
/** Preserve the complete upstream padded atom14/atom37 tensor layouts. */
export function expandAtoms(o, atoms, count) {
  if (count !== 14 && count !== 37) throw Error("Atom layout");
  const L = atoms.shape[0];
  return o.kernel(
    [L, count, 3],
    [atoms],
    `const r=Math.floor(i/${count * 3}),a=Math.floor(i/3)%${count},c=i%3;let src=a;${count === 14 ? "if(a===3)src=4;if(a===4)src=3;" : ""}o[i]=a<5?a0[r*15+src*3+c]:0;`,
    `let r=i/${count * 3}u;let a=i/3u%${count}u;let c=i%3u;var src=a;${count === 14 ? "if(a==3u){src=4u;}if(a==4u){src=3u;}" : ""}if(a<5u){o[i]=a0[r*15u+src*3u+c];}else{o[i]=0.0;}`,
  );
}
