import chemical from "./chemical-constants.js";
import { rigidFrames, normLast } from "./geometry.js";
export function backbone(o, xyz, alpha, sequence) {
  const L = xyz.shape[0],
    { R, T } = rigidFrames(o, xyz),
    base = o.tensor(
      [L, 4, 4],
      sequence.flatMap((aa) => chemical.base[aa].slice(0, 4).flat()),
    ),
    frames = o.tensor(
      [L, 4, 4],
      sequence.flatMap((aa) => chemical.torsionFrames[aa][2].flat()),
    ),
    psi = o.reshape(o.slice(alpha, 1, 2, 3), [L, 2]),
    angles = o.div(psi, o.add(normLast(o, psi), 1e-6));
  const rot = o.kernel(
      [L, 4, 4],
      [angles],
      `const n=Math.floor(i/16),k=i%16,c=a0[n*2],s=a0[n*2+1];o[i]=k===0||k===15?1:k===5||k===10?c:k===6?-s:k===9?s:0;`,
      `let n=i/16u;let k=i%16u;let c=a0[n*2u];let s=a0[n*2u+1u];var v=0.0;if(k==0u||k==15u){v=1.0;}else if(k==5u||k==10u){v=c;}else if(k==6u){v=-s;}else if(k==9u){v=s;}o[i]=v;`,
    ),
    oxygen = o.einsum(
      "nij,nj->ni",
      o.einsum("nij,njk->nik", frames, rot),
      o.reshape(o.slice(base, 1, 3, 4), [L, 4]),
    ),
    local = o.cat([o.slice(base, 1, 0, 3), o.reshape(oxygen, [L, 1, 4])], 1),
    local3 = o.slice(local, 2, 0, 3);
  return o.add(o.einsum("nij,naj->nai", R, local3), o.reshape(T, [L, 1, 3]));
}
