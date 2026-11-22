import { Ops } from "./ops.js";
import { Layers } from "./layers.js";
import { AtomModules } from "./encoders.js";
import { Trunk } from "./trunk.js";
import { sample } from "./sampler.js";
export class BoltzGen {
  constructor(backend, weights) {
    this.o = new Ops(backend, weights);
    this.layers = new Layers(this.o);
    this.atoms = new AtomModules(this.o, this.layers);
    this.trunk = new Trunk(this.o, this.layers);
  }
  async denoise(
    f,
    c,
    sInputs,
    sTrunk,
    noisy,
    sigma,
    { signal, onProgress, step },
  ) {
    const o = this.o,
      p = "structure_module.score_model",
      single = p + ".single_conditioner";
    let s = o.tidy(() => {
      let t = o.linear(
        o.norm(o.cat([sTrunk, sInputs]), single + ".norm_single"),
        single + ".single_embed",
      );
      const time = o.fill(
        [1, 1],
        Math.fround(Math.log(Math.fround(sigma / 16))) * 0.25,
      );
      const fourier = o.point(
        "cos",
        o.mul(o.linear(time, single + ".fourier_embed.proj"), 2 * Math.PI),
      );
      t = o.add(
        o.unsqueeze(
          o.linear(
            o.norm(fourier, single + ".norm_fourier"),
            single + ".fourier_to_single",
          ),
          1,
        ),
        t,
      );
      for (let i = 0; i < 2; i++)
        t = o.add(this.layers.transition(t, `${single}.transitions.${i}`), t);
      return t;
    });
    const denom = Math.fround(
        Math.sqrt(Math.fround(256 + Math.fround(sigma * sigma))),
      ),
      cin = Math.fround(1 / denom);
    const encoded = this.atoms.attentionEncoder(
      f,
      c.q,
      c.c,
      c.atom_enc_bias,
      p + ".atom_attention_encoder",
      o.mul(noisy, cin),
    );
    let a = o.add(encoded.a, o.seq(s, p + ".s_to_a_linear"));
    // Scope each token layer, so its temporary attention and MLP tensors are reusable.
    for (let i = 0; i < 24; i++) {
      signal?.throwIfAborted();
      a = this.layers.diffusionLayer(
        a,
        s,
        o.slice(c.token_trans_bias, -1, i * 16, i * 16 + 16),
        f.token_pad_mask,
        `${p}.token_transformer.layers.${i}`,
        16,
      );
      if (i % 4 === 3) await o.backend.flush?.();
    }
    a = o.norm(a, p + ".a_norm");
    const update = this.atoms.decoder(
      f,
      a,
      encoded.q,
      encoded.c,
      c.atom_dec_bias,
      p + ".atom_attention_decoder",
    );
    const skip = Math.fround(
        256 / Math.fround(Math.fround(sigma * sigma) + 256),
      ),
      out = Math.fround(Math.fround(sigma * 16) / denom);
    return o.add(o.mul(noisy, skip), o.mul(update, out));
  }
  async forward(
    f,
    random,
    { numSteps = 3, recyclingSteps = 0, signal, onProgress, stage } = {},
  ) {
    const o = this.o;
    let start = performance.now();
    const timings = [];
    const sInputs = this.atoms.input(f);
    await stage?.("input_embedder", sInputs);
    const { sInit, zInit, relative, pairMask } = o.tidy(() => {
      const relative = o.linear(f.relative_features, "rel_pos.linear_layer");
      let z = o.add(
        o.unsqueeze(o.linear(sInputs, "z_init_1"), 2),
        o.unsqueeze(o.linear(sInputs, "z_init_2"), 1),
      );
      z = o.add(z, relative);
      z = o.add(z, o.linear(f.token_bonds, "token_bonds"));
      z = o.add(z, o.embed(f.type_bonds, "token_bonds_type"));
      z = o.add(z, this.trunk.contact(f));
      return {
        sInit: o.linear(sInputs, "s_init"),
        zInit: z,
        relative,
        pairMask: o.mul(
          o.unsqueeze(f.token_pad_mask, 2),
          o.unsqueeze(f.token_pad_mask, 1),
        ),
      };
    });
    let s = o.fill(sInit.shape),
      z = o.fill(zInit.shape);
    const keep = () =>
      o.collect(f, random, sInputs, sInit, zInit, relative, pairMask, s, z);
    for (let recycle = 0; recycle <= recyclingSteps; recycle++) {
      s = o.add(sInit, o.linear(o.norm(s, "s_norm"), "s_recycle"));
      z = o.add(zInit, o.linear(o.norm(z, "z_norm"), "z_recycle"));
      const d = this.trunk.distance(f, z, relative, pairMask);
      await stage?.("token_distance_module", d);
      z = o.add(z, d);
      keep();
      const t = this.trunk.template(f, z, pairMask);
      await stage?.("template_module", t);
      z = o.add(z, t);
      keep();
      const msa = this.trunk.msa(f, z, sInputs, pairMask);
      await stage?.("msa_module", msa);
      z = o.add(z, msa);
      keep();
      for (let i = 0; i < 64; i++) {
        signal?.throwIfAborted();
        ({ s, z } = this.layers.pairformer(
          s,
          z,
          f.token_pad_mask,
          pairMask,
          `pairformer_module.layers.${i}`,
        ));
        await stage?.("pairformer_module.layers." + i, [s, z]);
        keep();
        if (i % 4 === 3) {
          await o.backend.flush?.();
          onProgress?.({ phase: "trunk", block: i + 1, total: 64, recycle });
        }
      }
    }
    const c = this.atoms.conditioning(f, s, z, relative);
    await stage?.("diffusion_conditioning", [
      c.q,
      c.c,
      c.atom_enc_bias,
      c.atom_dec_bias,
      c.token_trans_bias,
    ]);
    o.collect(f, random, sInputs, s, c);
    await o.backend.flush?.(true);
    timings.push({ phase: "trunk", milliseconds: performance.now() - start });
    start = performance.now();
    const result = await sample(this, f, c, sInputs, s, random, {
      numSteps,
      signal,
      onProgress,
      stage,
    });
    await o.backend.flush?.(true);
    timings.push({
      phase: "diffusion",
      milliseconds: performance.now() - start,
    });
    return { ...result, timings };
  }
}
