/** Dynamic Euler R3/SO3 flow step executed entirely as WGSL. */
export async function integrate(
  ops,
  x,
  pred,
  { dt, t, rotationScale, signal },
) {
  signal?.throwIfAborted();
  const L = x.length;
  try {
    ops.begin();
    const trans = ops.tensor(x.trans_t, [L, 3]),
      rot = ops.tensor(x.rotmats_t, [L, 9]),
      pt = ops.tensor(pred.pred_trans, [L, 3]),
      pr = ops.tensor(pred.pred_rotmats, [L, 9]);
    const a = ops.run(
      "eulerTranslation",
      [trans, pt],
      [L, 3],
      [dt, t],
      `o[idx]=a[idx]+((b[idx]-a[idx])/(1.0-pf(2)))*pf(1);`,
    );
    const b = ops.run(
      "eulerRotation",
      [rot, pr],
      [L, 9],
      [rotationScale],
      `
 let i=(idx/9u)*9u;let k=idx%9u;
 let R=mat3x3<f32>(vec3<f32>(a[i],a[i+3u],a[i+6u]),vec3<f32>(a[i+1u],a[i+4u],a[i+7u]),vec3<f32>(a[i+2u],a[i+5u],a[i+8u]));
 let P=mat3x3<f32>(vec3<f32>(b[i],b[i+3u],b[i+6u]),vec3<f32>(b[i+1u],b[i+4u],b[i+7u]),vec3<f32>(b[i+2u],b[i+5u],b[i+8u]));
 let M=transpose(R)*P;let I=mat3x3<f32>(vec3<f32>(1,0,0),vec3<f32>(0,1,0),vec3<f32>(0,0,1));
 var v=vec3<f32>(M[1][2]-M[2][1],M[2][0]-M[0][2],M[0][1]-M[1][0]);let sine=length(v)/2.0;let cosine=(M[0][0]+M[1][1]+M[2][2]-1.0)/2.0;let angle=atan2(sine,cosine);
 if(abs(angle-3.141592653589793)<=0.010031416){let outer=(I+M)*0.5;let rows=transpose(outer);var row=0u;for(var j=1u;j<3u;j++){if(length(rows[j])>length(rows[row])){row=j;}}v=sqrt(max(vec3<f32>(0),vec3<f32>(outer[0][0],outer[1][1],outer[2][2])))*angle*sign(rows[row]);}
 else if(angle<1e-8){v*=0.5/(1.0-angle*angle/6.0);}else{v*=angle/(2.0*sine);}
 v*=pf(1);let theta=length(v);var a1=1.0-theta*theta/6.0;var a2=0.5-theta*theta/24.0;if(theta>1e-7){a1=sin(theta)/theta;a2=(1.0-cos(theta))/(theta*theta);}
 let S=mat3x3<f32>(vec3<f32>(0,v.z,-v.y),vec3<f32>(-v.z,0,v.x),vec3<f32>(v.y,-v.x,0));let result=R*(I+a1*S+a2*(S*S));o[idx]=result[k%3u][k/3u];`,
    );
    const [trans_t, rotmats_t] = await ops.finish([a, b]);
    signal?.throwIfAborted();
    return { trans_t, rotmats_t };
  } finally {
    ops.cancel();
  }
}
