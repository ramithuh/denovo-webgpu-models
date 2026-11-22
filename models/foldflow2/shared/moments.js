// Cascade Welford moments matching the pinned ARM64 PyTorch CPU oracle.
// Four scalar lanes per vector and 16 vectors per chunk.
export function rowMoments(o, x) {
  const N = x.shape.at(-1),
    rows = x.size / N;
  if (N % 4) throw Error("Normalization width must be divisible by four");
  const chunks = Math.ceil(N / 64),
    slots = chunks * 4;
  const js = `const row=Math.floor(i/2),base=row*${N},means=new Float32Array(${slots}),vars=new Float32Array(${slots}),counts=new Int32Array(${chunks}),f=Math.fround;for(let ch=0;ch<${chunks};ch++){const count=Math.min(16,${N / 4}-ch*16);counts[ch]=count;for(let lane=0;lane<4;lane++){let mean=0,v=0;for(let j=0;j<count;j++){const z=a0[base+ch*64+j*4+lane],d=f(z-mean),c=f(1/(j+1));mean=rfFma(d,c,mean);v=rfFma(d,f(z-mean),v);}means[ch*4+lane]=mean;vars[ch*4+lane]=v;}}for(let stride=1;stride<${chunks};stride*=2){for(let a=0;a+stride<${chunks};a+=stride*2){const b=a+stride,n0=counts[a],n1=counts[b],c=f(n1/(n0+n1));for(let lane=0;lane<4;lane++){const p=a*4+lane,q=b*4+lane,d=f(means[q]-means[p]);means[p]=rfFma(c,d,means[p]);vars[p]=f(vars[p]+rfFma(f(f(d*d)*c),n0,vars[q]));}counts[a]+=n1;}}let count=0,mean=0,v=0;for(let lane=0;lane<4;lane++){const n1=counts[0],c=f(n1/(count+n1)),d=f(means[lane]-mean);mean=rfFma(c,d,mean);v=f(v+rfFma(f(f(d*d)*c),count,vars[lane]));count+=n1;}o[i]=i%2===0?mean:f(v/${N});`;
  const wg = `let row=i/2u;let base=row*${N}u;var means:array<f32,${slots}>;var vars:array<f32,${slots}>;var counts:array<u32,${chunks}>;for(var ch=0u;ch<${chunks}u;ch++){let count=min(16u,${N / 4}u-ch*16u);counts[ch]=count;for(var lane=0u;lane<4u;lane++){var mean=0.0;var v=0.0;for(var j=0u;j<count;j++){let z=a0[base+ch*64u+j*4u+lane];let d=z-mean;let c=rfdiv(1.0,f32(j+1u));mean=fma(d,c,mean);v=fma(d,z-mean,v);}means[ch*4u+lane]=mean;vars[ch*4u+lane]=v;}}for(var stride=1u;stride<${chunks}u;stride*=2u){for(var a=0u;a+stride<${chunks}u;a+=stride*2u){let b=a+stride;let n0=counts[a];let n1=counts[b];let c=rfdiv(f32(n1),f32(n0+n1));for(var lane=0u;lane<4u;lane++){let p=a*4u+lane;let q=b*4u+lane;let d=means[q]-means[p];means[p]=fma(c,d,means[p]);vars[p]=vars[p]+fma((d*d)*c,f32(n0),vars[q]);}counts[a]+=n1;}}var count=0u;var mean=0.0;var v=0.0;for(var lane=0u;lane<4u;lane++){let n1=counts[0];let c=rfdiv(f32(n1),f32(count+n1));let d=means[lane]-mean;mean=fma(c,d,mean);v=v+fma((d*d)*c,f32(count),vars[lane]);count+=n1;}o[i]=select(mean,v/${N}.0,i%2u==1u);`;
  return o.kernel([rows, 2], [x], js, wg);
}
export function normalization(o, x, w, b, group = false, eps = 1e-5) {
  const N = x.shape.at(-1),
    mom = rowMoments(o, x),
    inv =
      group && o.groupInverse
        ? o.groupInverse(mom, eps)
        : o.kernel(
            [x.size / N],
            [mom],
            group
              ? `o[i]=1/Math.sqrt(a0[i*2+1]+${eps});`
              : `o[i]=Math.fround(1/Math.fround(Math.sqrt(Math.fround(a0[i*2+1]+Math.fround(${eps})))));`,
            `o[i]=rfdiv(1.0,sqrt(a0[i*2u+1u]+${eps}));`,
          );
  return o.kernel(
    x.shape,
    [x, w, b, mom, inv],
    group
      ? `const row=Math.floor(i/${N}),k=i%${N},f=Math.fround,scale=f(a4[row]*a1[k]),bias=rfFma(-scale,a3[row*2],a2[k]);o[i]=rfFma(scale,a0[i],bias);`
      : `const row=Math.floor(i/${N}),k=i%${N},f=Math.fround,v=f(f(a0[i]-a3[row*2])*a4[row]);o[i]=rfFma(v,a1[k],a2[k]);`,
    group
      ? `let row=i/${N}u;let k=i%${N}u;let scale=a4[row]*a1[k];let bias=fma(-scale,a3[row*2u],a2[k]);o[i]=fma(scale,a0[i],bias);`
      : `let row=i/${N}u;let k=i%${N}u;let v=(a0[i]-a3[row*2u])*a4[row];o[i]=fma(v,a1[k],a2[k]);`,
  );
}
