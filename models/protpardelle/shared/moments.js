// FP32 cascade-Welford moments, matching the four-lane ARM reference reduction.
const f = Math.fround;
export function momentsWGSL(width, source = "b0") {
  if (width % 4) throw new Error("Vector moments require a multiple of four");
  const n = width / 4,
    chunks = Math.ceil(n / 16),
    depth = Math.max(1, Math.ceil(Math.log2(chunks)));
  const merge = (count, mean, square, level) => `{
    let old=mc[${level}];let total=old+${count};let factor=f32(${count})/f32(max(total,1u));
    let delta=${mean}-mm[${level}];let cd=delta*factor;
    ms[${level}]=fma(delta*f32(old),cd,ms[${level}]+${square});
    mm[${level}]+=cd;mc[${level}]=total;
  }`;
  return `var mc:array<u32,${depth}>;var mm:array<vec4<f32>,${depth}>;var ms:array<vec4<f32>,${depth}>;
  for(var chunk=0u;chunk<${chunks}u;chunk++){
    let count=min(16u,${n}u-chunk*16u);var m=vec4<f32>(0.0);var s=vec4<f32>(0.0);
    for(var j=0u;j<count;j++){
      let base=r*${width}u+chunk*64u+j*4u;
      let value=vec4<f32>(${source}[base],${source}[base+1u],${source}[base+2u],${source}[base+3u]);
      let delta=value-m;m=fma(vec4<f32>(1.0/f32(j+1u)),delta,m);s=fma(delta,value-m,s);
    }
    ${merge("count", "m", "s", "0")}
    var bits=chunk+1u;
    for(var level=1u;level<${depth}u && (bits&1u)==0u;level++){
      ${merge("mc[level-1u]", "mm[level-1u]", "ms[level-1u]", "level")}
      mc[level-1u]=0u;mm[level-1u]=vec4<f32>(0.0);ms[level-1u]=vec4<f32>(0.0);bits>>=1u;
    }
  }
  for(var level=1u;level<${depth}u;level++){${merge("mc[level]", "mm[level]", "ms[level]", "0")}}
  var mean=0.0;var square=0.0;var count=0u;
  for(var lane=0u;lane<4u;lane++){
    let total=count+${n}u;let factor=f32(${n})/f32(total);let delta=mm[0][lane]-mean;
    mean=fma(factor,delta,mean);square+=ms[0][lane]+((delta*delta)*factor)*f32(count);count=total;
  }
  let variance=square/f32(${width});`;
}
export function moments(values, offset, width) {
  const n = Math.floor(width / 4),
    chunks = Math.ceil(n / 16),
    depth = Math.ceil(Math.log2(chunks)),
    counts = new Int32Array(12),
    means = Array.from({ length: 12 }, () => new Float32Array(4)),
    squares = Array.from({ length: 12 }, () => new Float32Array(4));
  const merge = (count, mean, square, level) => {
    const old = counts[level],
      total = old + count,
      c = f(count / Math.max(total, 1));
    for (let k = 0; k < 4; k++) {
      const delta = f(mean[k] - means[level][k]),
        cd = f(c * delta);
      squares[level][k] = f(
        f(squares[level][k] + square[k]) + f(delta * old) * cd,
      );
      means[level][k] = f(means[level][k] + cd);
    }
    counts[level] = total;
  };
  for (let i = 0; i < chunks; i++) {
    const count = Math.min(16, n - i * 16),
      mean = new Float32Array(4),
      square = new Float32Array(4);
    for (let j = 0; j < count; j++)
      for (let k = 0; k < 4; k++) {
        const x = values[offset + i * 64 + j * 4 + k],
          delta = f(x - mean[k]);
        mean[k] = f(mean[k] + f(1 / (j + 1)) * delta);
        square[k] = f(square[k] + delta * f(x - mean[k]));
      }
    merge(count, mean, square, 0);
    let mask = i + 1;
    for (let j = 1; j < depth && (mask & 1) === 0; j++, mask >>= 1) {
      merge(counts[j - 1], means[j - 1], squares[j - 1], j);
      counts[j - 1] = 0;
      means[j - 1].fill(0);
      squares[j - 1].fill(0);
    }
  }
  for (let j = 1; j < depth; j++) merge(counts[j], means[j], squares[j], 0);
  let count = 0,
    mean = 0,
    square = 0;
  for (let j = n * 4; j < width; j++) {
    const x = values[offset + j],
      delta = f(x - mean);
    count++;
    mean = f(mean + f(delta / count));
    square = f(square + delta * f(x - mean));
  }
  for (let k = 0; k < 4; k++) {
    const total = count + n,
      c = f(n / Math.max(1, total)),
      delta = f(means[0][k] - mean);
    mean = f(mean + c * delta);
    square = f(square + f(squares[0][k] + f(f(f(delta * delta) * c) * count)));
    count = total;
  }
  return { mean, variance: f(square / width) };
}
