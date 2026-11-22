const geometry = `
fn ca(i:i32)->vec3<f32>{if(i<0 || i>=N){return vec3<f32>(0.0);}let j=u32(i)*111u+3u;return vec3<f32>(X[j],X[j+1u],X[j+2u]);}
fn dot3(a:vec3<f32>,b:vec3<f32>)->f32{return fma(a.z,b.z,fma(a.y,b.y,a.x*b.x));}
fn normalized(v:vec3<f32>)->vec3<f32>{let magnitude=max(sqrt(dot3(v,v)),1e-12);let q=v/magnitude;let residual=fma(-q,vec3<f32>(magnitude),v);return fma(residual,vec3<f32>(1.0/magnitude),q);}
fn direction(i:i32)->vec3<f32>{let v=ca(i+1)-ca(i);let d=sqrt(dot3(v,v));return normalized(v*select(0.0,1.0,d>3.6&&d<4.0));}
`;
export const graphOps = {
  fromArray(data, rows, cols) {
    return data?.buffer instanceof GPUBuffer
      ? data
      : this.tensor(rows, cols, data);
  },
  zeros(rows, cols) {
    return this.element(rows, cols, [this.zero], "b0[0]");
  },
  repeatRows(x, rows) {
    return this.element(rows, x.cols, [x], "b0[c]");
  },
  exp(x) {
    return this.element(x.rows, x.cols, [x], "exp(b0[i])");
  },
  gelu(x) {
    const out = this.tensor(x.rows, x.cols);
    this.dispatch(
      `let i=id.x;if(i>=${x.rows * x.cols}u){return;}let x=b0[i];let a=abs(x)*0.7071067811865476;let t=1.0/(1.0+0.3275911*a);let erf=sign(x)*(1.0-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*exp(-a*a));b1[i]=0.5*x*(1.0+erf);`,
      [x.buffer, out.buffer],
      [Math.ceil((x.rows * x.cols) / 64)],
    );
    return out;
  },
  logSoftmax(x) {
    const out = this.tensor(x.rows, x.cols),
      d = x.cols;
    this.dispatch(
      `let r=id.x;if(r>=${x.rows}u){return;}var largest=-3.402823e38;for(var j=0u;j<${d}u;j++){largest=max(largest,b0[r*${d}u+j]);}var sum=0.0;for(var j=0u;j<${d}u;j++){sum+=exp(b0[r*${d}u+j]-largest);}for(var j=0u;j<${d}u;j++){b1[r*${d}u+j]=b0[r*${d}u+j]-largest-log(sum);}`,
      [x.buffer, out.buffer],
      [Math.ceil(x.rows / 64)],
    );
    return out;
  },
  neighbors(node, edge, indices, center) {
    const d = node.cols,
      e = edge.cols,
      k = indices.cols,
      cols = e + d + (center ? d : 0),
      out = this.tensor(edge.rows, cols);
    this.dispatch(
      `let i=id.x;if(i>=${edge.rows * cols}u){return;}let r=i/${cols}u;let c=i%${cols}u;var v=0.0;${center ? `if(c<${d}u){v=b0[(r/${k}u)*${d}u+c];}else ` : ""}if(c<${(center ? d : 0) + e}u){v=b1[r*${e}u+c-${center ? d : 0}u];}else{v=b0[u32(b2[r])*${d}u+c-${(center ? d : 0) + e}u];}b3[i]=v;`,
      [node.buffer, edge.buffer, indices.buffer, out.buffer],
      [Math.ceil((edge.rows * cols) / 64)],
    );
    return out;
  },
  centerConcat(node, edge, indices) {
    const d = node.cols,
      e = edge.cols,
      k = indices.cols,
      cols = d + e,
      out = this.tensor(edge.rows, cols);
    this.dispatch(
      `let i=id.x;if(i>=${edge.rows * cols}u){return;}let r=i/${cols}u;let c=i%${cols}u;if(c<${d}u){b2[i]=b0[(r/${k}u)*${d}u+c];}else{b2[i]=b1[r*${e}u+c-${d}u];}`,
      [node.buffer, edge.buffer, out.buffer],
      [Math.ceil((edge.rows * cols) / 64)],
    );
    return out;
  },
  maskEdges(x, indices, mask) {
    return this.element(
      x.rows,
      x.cols,
      [x, this.input(mask, indices.rows)],
      `b0[i]*b1[r/${indices.cols}u]`,
    );
  },
  sumNeighbors(message, indices, mask, attend) {
    const n = indices.rows,
      k = indices.cols,
      d = message.cols,
      out = this.tensor(n, d);
    this.dispatch(
      `let i=id.x;if(i>=${n * d}u){return;}let r=i/${d}u;let c=i%${d}u;var sum=0.0;for(var j=0u;j<${k}u;j++){sum+=b0[(r*${k}u+j)*${d}u+c]*${attend ? `b2[r]*b2[u32(b1[r*${k}u+j])]` : "(1.0+0.0*b1[0]*b2[0])"};}b3[i]=sum/30.0;`,
      [message.buffer, indices.buffer, this.input(mask, n), out.buffer],
      [Math.ceil((n * d) / 64)],
    );
    return out;
  },
  graphFeatures(input, prefix) {
    const n = input.seqMask.length,
      k = Math.min(32, n),
      indices = this.tensor(n, k),
      frames = this.tensor(n, 9),
      distances = this.tensor(n, k),
      features = this.tensor(n * k, 167),
      coords = this.input(input.coords, n * 111),
      mask = this.input(input.seqMask, n);
    const nearest = `@group(0) @binding(0) var<storage,read> X:array<f32>;
@group(0) @binding(1) var<storage,read> M:array<f32>;
@group(0) @binding(2) var<storage,read> Replay:array<f32>;
@group(0) @binding(3) var<storage,read_write> I:array<f32>;
@group(0) @binding(4) var<storage,read_write> D:array<f32>;
const N:i32=${n};${geometry}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){let r=id.x;if(r>=${n}u){return;}var maxdist=0.0;var dist:array<f32,${n}>;for(var j=0u;j<${n}u;j++){let v=ca(i32(r))-ca(i32(j));dist[j]=sqrt(dot(v,v)+1e-6)*M[r]*M[j];maxdist=max(maxdist,dist[j]);}for(var j=0u;j<${n}u;j++){dist[j]+=(1.0-M[r]*M[j])*maxdist;}for(var a=0u;a<${k}u;a++){var best=0u;${input.neighborIndices ? "best=u32(Replay[r*" + k + "u+a]);" : "for(var j=1u;j<" + n + "u;j++){if(dist[j]<dist[best]){best=j;}}"}I[r*${k}u+a]=f32(best)+0.0*Replay[0];D[r*${k}u+a]=dist[best];dist[best]=3.402823e38;}}`;
    this.dispatch(
      nearest,
      [
        coords,
        mask,
        input.neighborIndices
          ? this.input(input.neighborIndices, n * k)
          : this.zero,
        indices.buffer,
        distances.buffer,
      ],
      [Math.ceil(n / 64)],
    );
    const orientation = `@group(0) @binding(0) var<storage,read> X:array<f32>;
@group(0) @binding(1) var<storage,read_write> O:array<f32>;
const N:i32=${n};${geometry}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){let r=id.x;if(r>=${n}u){return;}var a=vec3<f32>(0.0);var b=vec3<f32>(0.0);var c=vec3<f32>(0.0);if(r>=1u&&r<${Math.max(1, n - 2)}u){let u2=direction(i32(r)-1);let u1=direction(i32(r));a=normalized(u2-u1);b=normalized(cross(u2,u1));c=cross(a,b);}for(var j=0u;j<3u;j++){O[r*9u+j]=a[j];O[r*9u+3u+j]=b[j];O[r*9u+6u+j]=c[j];}}`;
    this.dispatch(orientation, [coords, frames.buffer], [Math.ceil(n / 64)]);
    // F32 positional embedding weights and bias packed into one binding.
    const w = this.weight(prefix + "embeddings.linear.weight"),
      b = this.weight(prefix + "embeddings.linear.bias"),
      position = this.allocate((16 * 66 + 16) * 4);
    this.encoder.copyBufferToBuffer(w.buffer, 0, position, 0, 16 * 66 * 4);
    this.encoder.copyBufferToBuffer(b.buffer, 0, position, 16 * 66 * 4, 16 * 4);
    const indexData = this.tensor(
      n,
      2,
      Float32Array.from({ length: n * 2 }, (_, i) =>
        i % 2
          ? input.seqMask[Math.floor(i / 2)]
          : input.residueIndex[Math.floor(i / 2)],
      ),
    );
    const feat = `@group(0) @binding(0) var<storage,read> X:array<f32>;
@group(0) @binding(1) var<storage,read> I:array<f32>;
@group(0) @binding(2) var<storage,read> D:array<f32>;
@group(0) @binding(3) var<storage,read> O:array<f32>;
@group(0) @binding(4) var<storage,read> P:array<f32>;
@group(0) @binding(5) var<storage,read> Ridx:array<f32>;
@group(0) @binding(6) var<storage,read_write> Y:array<f32>;
const N:i32=${n};${geometry}
fn frame(r:u32)->mat3x3<f32>{return transpose(mat3x3<f32>(vec3<f32>(O[r*9u],O[r*9u+1u],O[r*9u+2u]),vec3<f32>(O[r*9u+3u],O[r*9u+4u],O[r*9u+5u]),vec3<f32>(O[r*9u+6u],O[r*9u+7u],O[r*9u+8u])));}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){let t=id.x;if(t>=${n * k}u){return;}let i=t/${k}u;let j=u32(I[t]);let base=t*167u;let offset=select(65u,u32(clamp(Ridx[i*2u]-Ridx[j*2u]+32.0,0.0,64.0)),Ridx[i*2u+1u]==Ridx[j*2u+1u]);for(var c=0u;c<16u;c++){Y[base+c]=P[c*66u+offset]+P[1056u+c];}
let sa=array<i32,9>(0,-1,1,-1,-1,0,0,1,1);let sb=array<i32,9>(0,-1,1,0,1,-1,1,-1,0);
for(var p=0u;p<9u;p++){let diff=ca(i32(i)+sa[p])-ca(i32(j)+sb[p]);let dist=select(sqrt(dot(diff,diff)+1e-6),D[t],p==0u);for(var c=0u;c<16u;c++){let mu=2.0+20.0*f32(c)/15.0;let v=(dist-mu)/1.25;Y[base+16u+p*16u+c]=exp(-v*v);}}
let F=frame(i);let G=frame(j);let Ft=transpose(F);let delta=ca(i32(j))-ca(i32(i));let u=normalized(vec3<f32>(dot3(Ft[0],delta),dot3(Ft[1],delta),dot3(Ft[2],delta)));
let R=mat3x3<f32>(vec3<f32>(dot3(F[0],G[0]),dot3(F[1],G[0]),dot3(F[2],G[0])),vec3<f32>(dot3(F[0],G[1]),dot3(F[1],G[1]),dot3(F[2],G[1])),vec3<f32>(dot3(F[0],G[2]),dot3(F[1],G[2]),dot3(F[2],G[2])));
let diag=vec3<f32>(R[0][0],R[1][1],R[2][2]);let mags=0.5*sqrt(abs(vec3<f32>(1.0)+vec3<f32>(diag.x-diag.y-diag.z,-diag.x+diag.y-diag.z,-diag.x-diag.y+diag.z)));
let signs=sign(vec3<f32>(R[1][2]-R[2][1],R[2][0]-R[0][2],R[0][1]-R[1][0]));let w=sqrt(max(0.0,1.0+((diag.x+diag.y)+diag.z)))*0.5;let raw=vec4<f32>(signs*mags,w);let q=raw/max(length(raw),1e-12);
for(var c=0u;c<3u;c++){Y[base+160u+c]=u[c];}for(var c=0u;c<4u;c++){Y[base+163u+c]=q[c];}
}`;
    this.dispatch(
      feat,
      [
        coords,
        indices.buffer,
        distances.buffer,
        frames.buffer,
        position,
        indexData.buffer,
        features.buffer,
      ],
      [Math.ceil((n * k) / 64)],
    );
    return { features, indices };
  },
};
