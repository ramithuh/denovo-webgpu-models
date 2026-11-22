import { fmaF32 } from "./fma.js";
// Float32 exponential polynomial used by SLEEF 3.8.0 (Boost Software License).
// Reduction and FMA ordering follow PyTorch's ARM64 vector exp implementation.
export function expF32(x) {
  const f = Math.fround;
  if (x < -104) return 0;
  if (x > 88.72283905206835) return Infinity;
  const v = f(x * f(1.4426950408889634)),
    q = Math.round(v),
    s0 = fmaF32(q, f(-0.693145751953125), x),
    s = fmaF32(q, f(-1.428606765330187e-6), s0);
  let u = f(0.00019852761761285365);
  for (const c of [
    0.0013930435525253415, 0.008333360776305199, 0.041666485369205475,
    0.1666666716337204, 0.5,
  ])
    u = fmaF32(u, s, f(c));
  u = f(1 + fmaF32(f(s * s), u, s));
  return f(u * 2 ** q);
}
export const numericWGSL = `
// SLEEF u10 sine polynomial with double-float reduction on the time
// embedding domain |x| <= 2000. Float pairs preserve cancellation near pi.
fn rfDFAdd2(a:f32,b:f32)->vec2<f32>{
 let s=fma(1.0,a,b);let v=fma(-1.0,a,s);return vec2<f32>(s,fma(1.0,fma(-1.0,fma(-1.0,v,s),a),fma(-1.0,v,b)));
}
fn rfDFAdd(a:vec2<f32>,b:f32)->vec2<f32>{
 let s=fma(1.0,a.x,b);return vec2<f32>(s,fma(1.0,fma(1.0,fma(-1.0,s,a.x),b),a.y));
}
fn rfDFMul(a:vec2<f32>,b:vec2<f32>)->vec2<f32>{
 let p=fma(a.x,b.x,0.0);return vec2<f32>(p,fma(a.x,b.y,fma(a.y,b.x,fma(a.x,b.x,-p))));
}
fn rfReducedSin(t:vec2<f32>)->f32{
 let square=fma(t.x,t.x,0.0);
 let s=vec2<f32>(square,fma(t.x+t.x,t.y,fma(t.x,t.x,-square)));
 var u=2.6083159809786593541503e-6;
 u=fma(u,s.x,-0.0001981069071916863322258);
 u=fma(u,s.x,0.00833307858556509017944336);
 let c:f32=-0.166666597127914428710938;let p=fma(u,s.x,0.0);
 let cp=fma(1.0,c,p);let a=vec2<f32>(cp,fma(1.0,fma(-1.0,cp,c),p));
 let y=rfDFMul(a,s);
 let yh=fma(1.0,y.x,1.0);let x=vec2<f32>(yh,fma(1.0,fma(1.0,fma(-1.0,yh,1.0),y.x),y.y));
 return fma(t.x,x.x,fma(t.y,x.x,t.x*x.y));
}
fn rfTrigReduce(x:f32,q:f32,large:bool)->vec2<f32>{
 let a=fma(q,-1.57073974609375,0.0);
 var t=rfDFAdd2(x,a);
 t.y=fma(1.0,t.y,fma(q,-1.57073974609375,-a));
 let y=q*(-0.00005657970905303955);
 let next=rfDFAdd2(t.x,y);
 t=vec2<f32>(next.x,fma(1.0,next.y,t.y));
 if(large){t.y=t.y+fma(q,-0.00005657970905303955,-y);}
 t=rfDFAdd(t,q*(-9.92093629470503e-10));
 if(large){let h=fma(1.0,t.x,t.y);return vec2<f32>(h,fma(1.0,fma(-1.0,h,t.x),t.y));}
 return t;
}
fn rfSinCos(x:f32,large:bool)->vec2<f32>{
 let qs=round(x*0.3183098861837907);
 let qc=2.0*round(fma(x,0.3183098861837907,-0.5))+1.0;
 let sn=rfReducedSin(rfTrigReduce(x,2.0*qs,large));
 let cs=rfReducedSin(rfTrigReduce(x,qc,large));
 return vec2<f32>(select(sn,-sn,(i32(qs)&1)==1),select(cs,-cs,(i32(qc)&2)==0));
}
fn rfSin(x:f32)->f32{return rfSinCos(x,abs(x)>=125.0).x;}
fn rfCos(x:f32)->f32{return rfSinCos(x,abs(x)>=125.0).y;}
fn rfExp(x:f32)->f32 {
 if(x < -104.0){return 0.0;}
 let q=round(x*1.4426950408889634);
 let s0=fma(q,-0.693145751953125,x);
 let s=fma(q,-1.428606765330187e-6,s0);
 var u=0.00019852761761285365;
 u=fma(u,s,0.0013930435525253415);
 u=fma(u,s,0.008333360776305199);
 u=fma(u,s,0.041666485369205475);
 u=fma(u,s,0.1666666716337204);
 u=fma(u,s,0.5);
 u=1.0+fma(s*s,u,s);
 return ldexp(u,i32(q));
}
`;

// Correctly rounded division correction for normal finite FP32 operands.
export const divisionWGSL = `fn rfdiv(a:f32,b:f32)->f32 {
 let signBit=(bitcast<u32>(a)^bitcast<u32>(b))&0x80000000u;
 let x=abs(a);let y=abs(b);var bits=bitcast<u32>(x/y);
 if(bits==0u || bits>=0x7f800000u){return a/b;}
 for(var k=0u;k<4u;k++){
  let q=bitcast<f32>(bits);let e=fma(-q,y,x);
  let up=e>0.0;let nextBits=select(bits-1u,bits+1u,up);
  let halfGap=abs(bitcast<f32>(nextBits)-q)*0.5;
  let boundary=halfGap*y;
  if(abs(e)>boundary || (abs(e)==boundary && (bits&1u)==1u)){bits=nextBits;}else{break;}
 }
 return bitcast<f32>(bits|signBit);
}
`;
