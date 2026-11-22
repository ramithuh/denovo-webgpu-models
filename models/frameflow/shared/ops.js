import { matrixToQuat, quatToMatrix, linspaceF32 } from "./geometry.js";
import { decodeTensor } from "./weights.js";
const size = (s) => s.reduce((a, b) => a * b, 1);
export class Ops {
  constructor(weights) {
    this.weights = {};
    for (const [n, t] of Object.entries(weights))
      this.weights[n] = { data: decodeTensor(t), shape: t.shape };
  }
  tensor(data, shape) {
    return { data: Float32Array.from(data), shape };
  }
  weight(n) {
    const w = this.weights[n];
    if (!w) throw Error(`Missing weight ${n}`);
    return w;
  }
  collect() {}
  run(name, inputs, shape, p, code, fn) {
    const out = new Float32Array(size(shape));
    fn(out, ...inputs.map((t) => t.data));
    return { data: out, shape };
  }
  linear(x, name, relu = false) {
    const wn = name.endsWith("in_proj") ? name + "_weight" : name + ".weight",
      bn = name.endsWith("in_proj") ? name + "_bias" : name + ".bias";
    const w = this.weight(wn),
      bias = this.weight(bn),
      K = w.shape[1],
      N = w.shape[0],
      M = size(x.shape) / K;
    return this.run(
      "linear",
      [x, w, bias],
      [M, N],
      [M, K, N, +relu],
      ``,
      (o, a, b, c) => {
        for (let i = 0; i < M; i++)
          for (let j = 0; j < N; j++) {
            let s = 0;
            for (let k = 0; k < K; k++) s += a[i * K + k] * b[j * K + k];
            s += c[j];
            o[i * N + j] = relu ? Math.max(0, s) : s;
          }
      },
    );
  }
  add(a, b) {
    return this.run(
      "add",
      [a, b],
      a.shape,
      [],
      `o[idx]=a[idx]+b[idx];`,
      (o, x, y) => {
        for (let i = 0; i < o.length; i++) o[i] = x[i] + y[i];
      },
    );
  }
  norm(x, n) {
    const C = x.shape.at(-1),
      M = size(x.shape) / C;
    return this.run(
      "norm",
      [x, this.weight(n + ".weight"), this.weight(n + ".bias")],
      x.shape,
      [C],
      `
 let C=u(1);let row=idx/C;var mean=0.0;for(var k=0u;k<C;k++){mean+=a[row*C+k];}mean/=f32(C);var v=0.0;for(var k=0u;k<C;k++){let d=a[row*C+k]-mean;v+=d*d;}o[idx]=(a[idx]-mean)*inverseSqrt(v/f32(C)+0.00001)*b[idx%C]+c[idx%C];`,
      (o, a, b, c) => {
        for (let i = 0; i < M; i++) {
          let m = 0,
            v = 0;
          for (let k = 0; k < C; k++) m += a[i * C + k];
          m /= C;
          for (let k = 0; k < C; k++) v += (a[i * C + k] - m) ** 2;
          const f = 1 / Math.sqrt(v / C + 1e-5);
          for (let k = 0; k < C; k++)
            o[i * C + k] = (a[i * C + k] - m) * f * b[k] + c[k];
        }
      },
    );
  }
  mask(x, m, L, pair) {
    const C = x.shape.at(-1);
    return this.run(
      "mask",
      [x, m],
      x.shape,
      [C, L, +pair],
      `let row=idx/u(1);var v=b[row];if(u(3)>0u){v=b[row/u(2)]*b[row%u(2)];}o[idx]=a[idx]*v;`,
      (o, a, b) => {
        for (let i = 0; i < o.length; i++) {
          const r = Math.floor(i / C);
          o[i] = a[i] * (pair ? b[Math.floor(r / L)] * b[r % L] : b[r]);
        }
      },
    );
  }
  nodeFeatures(pos, mask, diff, P, D, t, r) {
    const L = pos.shape[0],
      C = P + 2 * D + 1;
    return this.run(
      "node",
      [pos, mask, diff],
      [L, C],
      [P, D, t, r],
      `
 let P=u(1);let D=u(2);let C=P+2u*D+1u;let row=idx/C;let k=idx%C;var v=0.0;
 if(k<P){let angle=a[row]*3.141592653589793/pow(2056.0,2.0*f32(k%(P/2u))/f32(P));v=select(cos(angle),sin(angle),k<P/2u)*b[row];}
 else if(k==P){v=c[row];}else{let j=(k-P-1u)%D;let t=select(f(4),f(3),k<P+1u+D);let freq=exp(f32(j%(D/2u))*(-log(2056.0)/f32(D/2u-1u)));let angle=(t*2056.0)*freq;v=select(cos(angle),sin(angle),j<D/2u)*b[row];}o[idx]=v;`,
      (o, a, b, c) => {
        for (let i = 0; i < L; i++) {
          for (let k = 0; k < P; k++) {
            const v = Math.fround(
              Math.fround(a[i] * Math.fround(Math.PI)) /
                Math.fround(2056 ** Math.fround((2 * (k % (P / 2))) / P)),
            );
            o[i * C + k] = (k < P / 2 ? Math.sin(v) : Math.cos(v)) * b[i];
          }
          o[i * C + P] = c[i];
          for (let z = 0; z < 2; z++)
            for (let k = 0; k < D; k++) {
              const freq = Math.fround(
                  Math.exp(
                    Math.fround(
                      (k % (D / 2)) *
                        Math.fround(-Math.log(2056) / (D / 2 - 1)),
                    ),
                  ),
                ),
                v = Math.fround(Math.fround((z ? r : t) * 2056) * freq);
              o[i * C + P + 1 + z * D + k] =
                (k < D / 2 ? Math.sin(v) : Math.cos(v)) * b[i];
            }
        }
      },
    );
  }
  relative(L, F) {
    return this.run(
      "relative",
      [],
      [L * L, F],
      [L, F],
      `let F=u(2);let row=idx/F;let k=idx%F;let d=f32(row/u(1))-f32(row%u(1));let angle=d*3.141592653589793/pow(2056.0,2.0*f32(k%(F/2u))/f32(F));o[idx]=select(cos(angle),sin(angle),k<F/2u);`,
      (o) => {
        for (let i = 0; i < L; i++)
          for (let j = 0; j < L; j++)
            for (let k = 0; k < F; k++) {
              const angle = Math.fround(
                Math.fround((i - j) * Math.fround(Math.PI)) /
                  Math.fround(2056 ** Math.fround((2 * (k % (F / 2))) / F)),
              );
              o[(i * L + j) * F + k] =
                k < F / 2 ? Math.sin(angle) : Math.cos(angle);
            }
      },
    );
  }
  edgeFeatures(sp, rp, t, sc, dm, L, F, B, embed) {
    const C = 3 * F + 2 * B + (embed ? 2 : 0);
    return this.run(
      "edgeFeatures",
      [sp, rp, t, sc, dm, this.tensor(linspaceF32(0.001, 20, B), [B])],
      [L * L, C],
      [L, F, B, +embed],
      `
 let L=u(1);let F=u(2);let B=u(3);let C=3u*F+2u*B+2u*u(4);let row=idx/C;let k=idx%C;let i=row/L;let j=row%L;var v=0.0;
 if(k<F){v=a[i*F+k];}else if(k<2u*F){v=a[j*F+k-F];}else if(k<3u*F){v=b[row*F+k-2u*F];}
 else if(k<3u*F+2u*B){let bin=(k-3u*F)%B;var d2=0.0;for(var h=0u;h<3u;h++){var z=c[i*3u+h]-c[j*3u+h];if(k>=3u*F+B){z=d[i*3u+h]-d[j*3u+h];}d2+=z*z;}let lo=f[bin];var hi=1e8;if(bin+1u<B){hi=f[bin+1u];}v=select(0.0,1.0,sqrt(d2)>lo && sqrt(d2)<hi);}
 else{v=select(e[j],e[i],k==3u*F+2u*B);}o[idx]=v;`,
      (o, a, b, c, d, e, bounds) => {
        for (let i = 0; i < L; i++)
          for (let j = 0; j < L; j++) {
            let row = i * L + j;
            for (let k = 0; k < F; k++) {
              o[row * C + k] = a[i * F + k];
              o[row * C + F + k] = a[j * F + k];
              o[row * C + 2 * F + k] = b[row * F + k];
            }
            for (let z = 0; z < 2; z++) {
              const x = z ? d : c;
              const dis = Math.hypot(
                x[i * 3] - x[j * 3],
                x[i * 3 + 1] - x[j * 3 + 1],
                x[i * 3 + 2] - x[j * 3 + 2],
              );
              for (let k = 0; k < B; k++)
                o[row * C + 3 * F + z * B + k] = +(
                  dis > bounds[k] && dis < (k === B - 1 ? 1e8 : bounds[k + 1])
                );
            }
            if (embed) {
              o[row * C + 3 * F + 2 * B] = e[i];
              o[row * C + 3 * F + 2 * B + 1] = e[j];
            }
          }
      },
    );
  }
  pairConcat(z, s, L) {
    const E = z.shape[1],
      C = s.shape[1],
      N = E + 2 * C;
    return this.run(
      "pairConcat",
      [z, s],
      [L * L, N],
      [L, E, C],
      `let E=u(2);let C=u(3);let N=E+2u*C;let row=idx/N;let k=idx%N;var v=0.0;if(k<E){v=a[row*E+k];}else if(k<E+C){v=b[(row/u(1))*C+k-E];}else{v=b[(row%u(1))*C+k-E-C];}o[idx]=v;`,
      (o, a, b) => {
        for (let i = 0; i < L * L; i++) {
          o.set(a.subarray(i * E, (i + 1) * E), i * N);
          o.set(
            b.subarray(Math.floor(i / L) * C, (Math.floor(i / L) + 1) * C),
            i * N + E,
          );
          o.set(b.subarray((i % L) * C, ((i % L) + 1) * C), i * N + E + C);
        }
      },
    );
  }
  frames(rot, trans) {
    const L = trans.shape[0];
    return this.run(
      "frames",
      [rot, trans],
      [L, 7],
      [],
      `let row=idx/7u;let k=idx%7u;if(k<4u){let q=matquat(row*9u);o[idx]=q[k];}else{o[idx]=b[row*3u+k-4u]*0.1;}`,
      (o, a, b) => {
        for (let i = 0; i < L; i++) {
          o.set(matrixToQuat(a.subarray(i * 9, i * 9 + 9)), i * 7);
          for (let k = 0; k < 3; k++) o[i * 7 + 4 + k] = b[i * 3 + k] * 0.1;
        }
      },
    );
  }
  points(x, frames, P) {
    const L = x.shape[0];
    return this.run(
      "points",
      [x, frames],
      [L, P, 3],
      [P],
      `let P=u(1);let row=idx/(3u*P);let point=(idx/3u)%P;let c=idx%3u;let R=qmat(vec4<f32>(b[row*7u],b[row*7u+1u],b[row*7u+2u],b[row*7u+3u]));var v=b[row*7u+4u+c];for(var k=0u;k<3u;k++){v+=R[k][c]*a[row*P*3u+k*P+point];}o[idx]=v;`,
      (o, a, b) => {
        for (let i = 0; i < L; i++) {
          const r = quatToMatrix(b.subarray(i * 7, i * 7 + 4));
          for (let p = 0; p < P; p++)
            for (let c = 0; c < 3; c++) {
              let v = b[i * 7 + 4 + c];
              for (let k = 0; k < 3; k++)
                v += r[c * 3 + k] * a[i * P * 3 + k * P + p];
              o[(i * P + p) * 3 + c] = v;
            }
        }
      },
    );
  }
  ipaLogits(q, kv, qp, kp, bias, heads, mask, L, H, K, Q, V) {
    return this.run(
      "ipaLogits",
      [q, kv, qp, kp, bias, heads, mask],
      [H * L, L],
      [L, H, K, Q, V],
      `
 let L=u(1);let H=u(2);let K=u(3);let Q=u(4);let V=u(5);let j=idx%L;let i=(idx/L)%L;let h=idx/(L*L);var dot=0.0;for(var k=0u;k<K;k++){dot+=a[(i*H+h)*K+k]*b[(j*H+h)*2u*K+k];}var dist=0.0;let weight=log(1.0+exp(f[h]))*sqrt(1.0/(3.0*(f32(Q)*4.5)));for(var p=0u;p<Q;p++){var v=0.0;for(var k=0u;k<3u;k++){let diff=c[((i*H+h)*Q+p)*3u+k]-d[((j*H+h)*(Q+V)+p)*3u+k];v+=diff*diff;}dist+=v*weight;}o[idx]=dot*sqrt(1.0/(3.0*f32(K)))+e[(i*L+j)*H+h]*sqrt(1.0/3.0)-0.5*dist+100000.0*(g[i]*g[j]-1.0);`,
      (o, a, b, c, d, e, f, g) => {
        for (let h = 0; h < H; h++)
          for (let i = 0; i < L; i++)
            for (let j = 0; j < L; j++) {
              let dot = 0,
                dist = 0;
              for (let k = 0; k < K; k++)
                dot += a[(i * H + h) * K + k] * b[(j * H + h) * 2 * K + k];
              for (let p = 0; p < Q; p++)
                for (let k = 0; k < 3; k++)
                  dist +=
                    (c[((i * H + h) * Q + p) * 3 + k] -
                      d[((j * H + h) * (Q + V) + p) * 3 + k]) **
                    2;
              const weight =
                Math.log1p(Math.exp(f[h])) * Math.sqrt(1 / (3 * Q * 4.5));
              o[(h * L + i) * L + j] =
                dot * Math.sqrt(1 / (3 * K)) +
                e[(i * L + j) * H + h] * Math.sqrt(1 / 3) -
                0.5 * dist * weight +
                1e5 * (g[i] * g[j] - 1);
            }
      },
    );
  }
  softmax(x, N) {
    return this.run(
      "softmax",
      [x],
      x.shape,
      [N],
      `let N=u(1);let row=idx/N;var m=-3.4e38;for(var j=0u;j<N;j++){m=max(m,a[row*N+j]);}var s=0.0;for(var j=0u;j<N;j++){s+=exp(a[row*N+j]-m);}o[idx]=exp(a[idx]-m)/s;`,
      (o, a) => {
        for (let i = 0; i < a.length; i += N) {
          let m = -Infinity,
            s = 0;
          for (let j = 0; j < N; j++) m = Math.max(m, a[i + j]);
          for (let j = 0; j < N; j++) s += Math.exp(a[i + j] - m);
          for (let j = 0; j < N; j++) o[i + j] = Math.exp(a[i + j] - m) / s;
        }
      },
    );
  }
  ipaOutput(att, kv, kp, dz, frames, L, H, K, Q, V, Z) {
    const C = H * (K + 4 * V + Z);
    return this.run(
      "ipaOutput",
      [att, kv, kp, dz, frames],
      [L, C],
      [L, H, K, Q, V, Z],
      `
 let L=u(1);let H=u(2);let K=u(3);let Q=u(4);let V=u(5);let Z=u(6);let C=H*(K+4u*V+Z);let i=idx/C;let k=idx%C;var v=0.0;
 if(k<H*K){let h=k/K;let ch=k%K;for(var j=0u;j<L;j++){v+=a[(h*L+i)*L+j]*b[(j*H+h)*2u*K+K+ch];}}
 else if(k<H*K+4u*H*V){let z=k-H*K;let typ=z/(H*V);let h=(z%(H*V))/V;let point=z%V;var pt=vec3<f32>(0.0);for(var j=0u;j<L;j++){let base=((j*H+h)*(Q+V)+Q+point)*3u;pt+=a[(h*L+i)*L+j]*vec3<f32>(c[base],c[base+1u],c[base+2u]);}pt-=vec3<f32>(e[i*7u+4u],e[i*7u+5u],e[i*7u+6u]);let R=qmat(vec4<f32>(e[i*7u],e[i*7u+1u],e[i*7u+2u],e[i*7u+3u]));pt=transpose(R)*pt;if(typ==3u){v=sqrt(dot(pt,pt)+1e-8);}else{v=pt[typ];}}
 else{let z=k-H*K-4u*H*V;let h=z/Z;let ch=z%Z;for(var j=0u;j<L;j++){v+=a[(h*L+i)*L+j]*d[(i*L+j)*Z+ch];}}o[idx]=v;`,
      (o, a, b, c, d, e) => {
        for (let i = 0; i < L; i++) {
          const r = quatToMatrix(e.subarray(i * 7, i * 7 + 4));
          for (let h = 0; h < H; h++) {
            for (let k = 0; k < K; k++) {
              let v = 0;
              for (let j = 0; j < L; j++)
                v += a[(h * L + i) * L + j] * b[(j * H + h) * 2 * K + K + k];
              o[i * C + h * K + k] = v;
            }
            for (let p = 0; p < V; p++) {
              const pt = [0, 0, 0];
              for (let j = 0; j < L; j++)
                for (let k = 0; k < 3; k++)
                  pt[k] +=
                    a[(h * L + i) * L + j] *
                    c[((j * H + h) * (Q + V) + Q + p) * 3 + k];
              for (let k = 0; k < 3; k++) pt[k] -= e[i * 7 + 4 + k];
              let norm = 0;
              for (let k = 0; k < 3; k++) {
                let v = 0;
                for (let j = 0; j < 3; j++) v += r[j * 3 + k] * pt[j];
                o[i * C + H * K + k * H * V + h * V + p] = v;
                norm += v * v;
              }
              o[i * C + H * K + 3 * H * V + h * V + p] = Math.sqrt(norm + 1e-8);
            }
            for (let k = 0; k < Z; k++) {
              let v = 0;
              for (let j = 0; j < L; j++)
                v += a[(h * L + i) * L + j] * d[(i * L + j) * Z + k];
              o[i * C + H * K + 4 * H * V + h * Z + k] = v;
            }
          }
        }
      },
    );
  }
  attentionLogits(qkv, mask, L, C, H) {
    return this.run(
      "attentionLogits",
      [qkv, mask],
      [H * L, L],
      [L, C, H],
      `let L=u(1);let C=u(2);let H=u(3);let D=C/H;let j=idx%L;let i=(idx/L)%L;let h=idx/(L*L);var v=0.0;for(var k=0u;k<D;k++){v+=a[i*C*3u+h*D+k]*a[j*C*3u+C+h*D+k];}o[idx]=select(-3.4e38,v/sqrt(f32(D)),b[j]>0.0);`,
      (o, a, b) => {
        const D = C / H;
        for (let h = 0; h < H; h++)
          for (let i = 0; i < L; i++)
            for (let j = 0; j < L; j++) {
              let v = 0;
              for (let k = 0; k < D; k++)
                v += a[i * C * 3 + h * D + k] * a[j * C * 3 + C + h * D + k];
              o[(h * L + i) * L + j] = b[j] ? v / Math.sqrt(D) : -3.4e38;
            }
      },
    );
  }
  attentionOutput(att, qkv, L, C, H) {
    return this.run(
      "attentionOutput",
      [att, qkv],
      [L, C],
      [L, C, H],
      `let L=u(1);let C=u(2);let D=C/u(3);let i=idx/C;let k=idx%C;let h=k/D;var v=0.0;for(var j=0u;j<L;j++){v+=a[(h*L+i)*L+j]*b[j*C*3u+C*2u+k];}o[idx]=v;`,
      (o, a, b) => {
        for (let i = 0; i < L; i++)
          for (let k = 0; k < C; k++) {
            let v = 0;
            const h = Math.floor(k / (C / H));
            for (let j = 0; j < L; j++)
              v += a[(h * L + i) * L + j] * b[j * C * 3 + C * 2 + k];
            o[i * C + k] = v;
          }
      },
    );
  }
  updateFrames(frames, delta, mask, diff) {
    const L = frames.shape[0];
    return this.run(
      "updateFrames",
      [frames, delta, mask, diff],
      [L, 7],
      [],
      `
 let i=idx/7u;let k=idx%7u;let q=vec4<f32>(a[i*7u],a[i*7u+1u],a[i*7u+2u],a[i*7u+3u]);let m=c[i]*d[i];let v=vec3<f32>(b[i*6u],b[i*6u+1u],b[i*6u+2u])*m;
 if(k<4u){let r=normalize(q+vec4<f32>(-dot(q.yzw,v),q.x*v+cross(q.yzw,v)));o[idx]=r[k];}else{let dt=qmat(q)*vec3<f32>(b[i*6u+3u],b[i*6u+4u],b[i*6u+5u]);o[idx]=a[idx]+dt[k-4u]*m;}`,
      (o, a, b, c, d) => {
        for (let i = 0; i < L; i++) {
          const q = a.subarray(i * 7, i * 7 + 4),
            r = quatToMatrix(q),
            m = c[i] * d[i],
            x = b[i * 6] * m,
            y = b[i * 6 + 1] * m,
            z = b[i * 6 + 2] * m;
          const nq = [
              q[0] - q[1] * x - q[2] * y - q[3] * z,
              q[1] + q[0] * x + q[2] * z - q[3] * y,
              q[2] + q[0] * y + q[3] * x - q[1] * z,
              q[3] + q[0] * z + q[1] * y - q[2] * x,
            ],
            norm = Math.hypot(...nq);
          o.set(
            nq.map((v) => v / norm),
            i * 7,
          );
          for (let k = 0; k < 3; k++) {
            let v = 0;
            for (let j = 0; j < 3; j++) v += r[k * 3 + j] * b[i * 6 + 3 + j];
            o[i * 7 + 4 + k] = a[i * 7 + 4 + k] + v * m;
          }
        }
      },
    );
  }
  frameOutput(frames, rot) {
    const L = frames.shape[0],
      C = rot ? 9 : 3;
    return this.run(
      "frameOutput",
      [frames],
      [L, C],
      [+rot],
      `let C=select(3u,9u,u(1)>0u);let i=idx/C;let k=idx%C;if(u(1)>0u){let R=qmat(vec4<f32>(a[i*7u],a[i*7u+1u],a[i*7u+2u],a[i*7u+3u]));o[idx]=R[k%3u][k/3u];}else{o[idx]=a[i*7u+4u+k]*10.0;}`,
      (o, a) => {
        for (let i = 0; i < L; i++)
          if (rot) o.set(quatToMatrix(a.subarray(i * 7, i * 7 + 4)), 9 * i);
          else for (let j = 0; j < 3; j++) o[i * 3 + j] = a[i * 7 + 4 + j] * 10;
      },
    );
  }
}
