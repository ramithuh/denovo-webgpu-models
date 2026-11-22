import { fmaF32 } from "../shared/fma.js";
import { expF32, scalarExpF32, erfF32 } from "../shared/numeric.js";
import { Algebra, sizeOf } from "../shared/tensor.js";
export class CPUAlgebra extends Algebra {
  constructor() {
    super();
    this.kernels = new Map();
    this.resident = new Set();
  }
  upload(shape, data, persistent = false) {
    if (persistent) this.resident.add(data);
    return { shape, size: sizeOf(shape), data };
  }
  linear(x, w, b) {
    const K = x.shape.at(-1),
      N = w.shape[0],
      M = x.size / K,
      y = new Float32Array(M * N),
      a = x.data,
      v = w.data;
    if (w.shape[1] !== K) throw Error("Linear shape mismatch");
    for (let r = 0; r < M; r++)
      for (let n = 0; n < N; n++) {
        const ai = r * K,
          wi = n * K;
        // PyTorch addmm accumulates from the bias, with F32 fused multiply-add.
        let sum = b ? b.data[n] : 0;
        for (let k = 0; k < K; k++)
          sum = fmaF32(a[ai + k], v[wi + k], sum);
        y[r * N + n] = sum;
      }
    return { shape: [...x.shape.slice(0, -1), N], size: y.length, data: y };
  }
  kernel(shape, inputs, js) {
    const size = sizeOf(shape);
    let fn = this.kernels.get(js);
    if (!fn) {
      fn = new Function(
        "rfExp",
        "rfScalarExp",
        "rfFma",
        "rfErf",
        "o",
        ...inputs.map((_, i) => `a${i}`),
        `const rfdiv=(a,b)=>a/b;for(let i=0;i<o.length;i++){${js}}`,
      );
      this.kernels.set(js, fn);
    }
    const data = new Float32Array(size);
    fn(expF32, scalarExpF32, fmaF32, erfF32, data, ...inputs.map((t) => t.data));
    return { shape, size, data };
  }
  async read(t) {
    return t.data.slice();
  }
  async checkpoint() {}
  clear() {}
  destroy() {
    this.kernels.clear();
    this.resident.clear();
  }
  memoryUsage() {
    return {
      bytes: Array.from(this.resident).reduce((s, x) => s + x.byteLength, 0),
      transientBuffers: 0,
    };
  }
}
