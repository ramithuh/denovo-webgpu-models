import { fmaF32 } from "../shared/fma.js";
import { expF32, scalarExpF32 } from "../shared/numeric.js";
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
  kernel(shape, inputs, js) {
    const size = sizeOf(shape);
    let fn = this.kernels.get(js);
    if (!fn) {
      fn = new Function(
        "rfExp",
        "rfScalarExp",
        "rfFma",
        "o",
        ...inputs.map((_, i) => `a${i}`),
        `const rfdiv=(a,b)=>a/b;for(let i=0;i<o.length;i++){${js}}`,
      );
      this.kernels.set(js, fn);
    }
    const data = new Float32Array(size);
    fn(expF32, scalarExpF32, fmaF32, data, ...inputs.map((t) => t.data));
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
