import { parseStructure, ALPHABET } from "./parser.js";
import { featurize, rawFeatures } from "./features.js";
export function seededRandom(seed = 42) {
  let s = seed >>> 0;
  return () => {
    s += 0x6d2b79f5;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function normalDraws(L, rng) {
  return Float32Array.from(
    { length: L },
    () =>
      Math.sqrt(-2 * Math.log(Math.max(1e-12, rng()))) *
      Math.cos(2 * Math.PI * rng()),
  );
}
function softmax(logits, temperature = 1, bias = null, n = logits.length) {
  let max = -Infinity;
  for (let i = 0; i < n; i++)
    max = Math.max(max, (logits[i] + (bias?.[i] ?? 0)) / temperature);
  const p = Float32Array.from({ length: n }, (_, i) =>
    Math.exp((logits[i] + (bias?.[i] ?? 0)) / temperature - max),
  );
  const sum = p.reduce((a, b) => a + b, 0);
  return p.map((v) => v / sum);
}
function logSoftmax(x) {
  const max = Math.max(...x),
    ln = Math.log(x.reduce((s, v) => s + Math.exp(v - max), 0)) + max;
  return Float32Array.from(x, (v) => v - ln);
}
export class LigandModel {
  constructor(ops, bundle) {
    this.o = ops;
    this.manifest = bundle.manifest;
    this.w = Object.fromEntries(
      Object.entries(bundle.weights).map(([k, t]) => [
        k,
        ops.keep(ops.tensor(t.data, t.shape)),
      ]),
    );
    ops.sealWeights?.();
    this.trace = {};
    this.count = {};
  }
  start(trace) {
    this.tracing = trace;
    this.trace = {};
    this.count = {};
  }
  save(name, t) {
    if (this.tracing) this.trace[name] = this.o.keep(t);
    return t;
  }
  call(name, t) {
    const n = this.count[name] ?? 0;
    this.count[name] = n + 1;
    return this.save(name + "." + n, t);
  }
  linear(x, p) {
    return this.o.linear(x, this.w[p + ".weight"], this.w[p + ".bias"]);
  }
  norm(x, p) {
    return this.o.norm(x, this.w[p + ".weight"], this.w[p + ".bias"]);
  }
  mlp(x, p, keys = ["W1", "W2", "W3"]) {
    return this.linear(
      this.o.gelu(
        this.linear(
          this.o.gelu(this.linear(x, p + "." + keys[0])),
          p + "." + keys[1],
        ),
      ),
      p + "." + keys[2],
    );
  }
  ff(v, p) {
    return this.linear(
      this.o.gelu(this.linear(v, p + ".dense.W_in")),
      p + ".dense.W_out",
    );
  }
  dec(v, e, p, K, mask, attend) {
    return this.o.scope(() => {
      const o = this.o,
        rows = v.shape[0],
        repeat = Int32Array.from({ length: rows * K }, (_, i) =>
          Math.floor(i / K),
        );
      let msg = this.mlp(o.concat(o.gather(v, repeat), e), p);
      if (attend) msg = o.mask(msg, attend);
      v = this.norm(o.add(v, o.reduce(msg, K)), p + ".norm1");
      v = this.norm(o.add(v, this.ff(v, p)), p + ".norm2");
      if (mask) v = o.mask(v, mask);
      return this.call(p, v);
    });
  }
  enc(v, e, p, idx, K, mask, attend) {
    return this.o.scope(() => {
      const o = this.o,
        repeat = Int32Array.from({ length: idx.length }, (_, i) =>
          Math.floor(i / K),
        );
      const cat = () => o.concat(o.gather(v, repeat), e, o.gather(v, idx));
      let msg = o.mask(this.mlp(cat(), p), attend);
      v = this.norm(o.add(v, o.reduce(msg, K)), p + ".norm1");
      v = o.mask(this.norm(o.add(v, this.ff(v, p)), p + ".norm2"), mask);
      e = this.norm(
        o.add(e, this.mlp(cat(), p, ["W11", "W12", "W13"])),
        p + ".norm3",
      );
      this.save(p + ".0.h_V", v);
      this.save(p + ".0.h_E", e);
      return [v, e];
    });
  }
  encode(input) {
    return this.o.scope(() => {
      const o = this.o,
        { L, K, M, E_idx, Y_m, mask } = input,
        raw = o.rawFeatures ? o.rawFeatures(input) : rawFeatures(input),
        t = (data, R, D) => (data?.shape ? data : o.tensor(data, [R, D]));
      const pos = this.linear(
        t(raw.positional, L * K, 66),
        "features.embeddings.linear",
      );
      const edgeRaw = this.save(
        "geometry.edge_embedding",
        o.concat(pos, t(raw.rbf, L * K, 400)),
      );
      const E = this.save(
        "features.0.E",
        this.norm(
          this.linear(edgeRaw, "features.edge_embedding"),
          "features.norm_edges",
        ),
      );
      const types = t(raw.types, L * M, 147),
        typeProjection = this.linear(types, "features.type_linear");
      const nodeRaw = this.save(
        "geometry.node_project_down",
        o.concat(
          t(raw.distances, L * M, 80),
          typeProjection,
          t(raw.angles, L * M, 4),
        ),
      );
      const V = this.save(
        "features.0.V",
        this.norm(
          this.linear(nodeRaw, "features.node_project_down"),
          "features.norm_nodes",
        ),
      );
      const yRaw = this.save("geometry.y_edges", t(raw.yRbf, L * M * M, 16));
      this.save("geometry.y_nodes", types);
      const Yn = this.save(
        "features.0.Y_nodes",
        this.norm(
          this.linear(types, "features.y_nodes"),
          "features.norm_y_nodes",
        ),
      );
      const Ye = this.save(
        "features.0.Y_edges",
        this.norm(
          this.linear(yRaw, "features.y_edges"),
          "features.norm_y_edges",
        ),
      );
      let v = o.zeros([L, 128]),
        e = this.linear(E, "W_e");
      const ec = this.linear(V, "W_v"),
        maskT = t(mask, L, 1),
        attend = t(
          Float32Array.from(E_idx, (j, r) => mask[Math.floor(r / K)] * mask[j]),
          L * K,
          1,
        );
      for (let i = 0; i < 3; i++)
        [v, e] = this.enc(v, e, "encoder_layers." + i, E_idx, K, maskT, attend);
      let vc = this.linear(v, "W_c"),
        yn = this.linear(Yn, "W_nodes_y"),
        ye = this.linear(Ye, "W_edges_y");
      const ym = t(Y_m, L * M, 1),
        ymEdges = t(
          Float32Array.from(
            { length: L * M * M },
            (_, i) =>
              Y_m[Math.floor(i / M)] *
              Y_m[Math.floor(i / (M * M)) * M + (i % M)],
          ),
          L * M * M,
          1,
        );
      for (let i = 0; i < 2; i++) {
        yn = this.dec(yn, ye, "y_context_encoder_layers." + i, M, ym, ymEdges);
        vc = this.dec(
          vc,
          o.concat(ec, yn),
          "context_encoder_layers." + i,
          M,
          maskT,
          ym,
        );
      }
      v = o.add(
        v,
        this.call("V_C_norm", this.norm(this.linear(vc, "V_C"), "V_C_norm")),
      );
      this.save("encoded.h_V", v);
      this.save("encoded.h_E", e);
      return { v, e };
    });
  }
  order(input, randn) {
    const { L, mask, chain_mask, ties } = input;
    const order = Array.from({ length: L }, (_, i) => i).sort(
        (a, b) =>
          (chain_mask[a] * mask[a] + 0.0001) * Math.abs(randn[a]) -
            (chain_mask[b] * mask[b] + 0.0001) * Math.abs(randn[b]) || a - b,
      ),
      groups = [],
      seen = new Set();
    for (const i of order) {
      if (seen.has(i)) continue;
      const tie = ties.find((t) => t.ids.includes(i));
      const group = tie ?? { ids: [i], weights: [1] };
      groups.push(group);
      group.ids.forEach((j) => seen.add(j));
    }
    const flat = groups.flatMap((g) => g.ids),
      rank = Int32Array.from({ length: L }, (_, i) => flat.indexOf(i));
    return { groups, flat, rank };
  }
  async score(input, { randn, trace = false } = {}) {
    this.start(trace);
    const o = this.o,
      { L, K, E_idx, mask, S } = input;
    const order = this.order(input, randn ?? normalDraws(L, seededRandom()));
    let { v, e } = this.encode(input);
    const bw = o.tensor(
      Float32Array.from(
        E_idx,
        (j, r) =>
          mask[Math.floor(r / K)] *
          (order.rank[j] < order.rank[Math.floor(r / K)]),
      ),
      [L * K, 1],
    );
    const fw = o.tensor(
      Float32Array.from(
        E_idx,
        (j, r) =>
          mask[Math.floor(r / K)] *
          (order.rank[j] >= order.rank[Math.floor(r / K)]),
      ),
      [L * K, 1],
    );
    const hs = o.gather(this.w["W_s.weight"], S),
      es = o.concat(e, o.gather(hs, E_idx)),
      enc = o.mask(o.concat(e, o.zeros([L * K, 128]), o.gather(v, E_idx)), fw),
      maskT = o.tensor(mask, [L, 1]);
    for (let l = 0; l < 3; l++)
      v = this.dec(
        v,
        o.add(o.mask(o.concat(es, o.gather(v, E_idx)), bw), enc),
        "decoder_layers." + l,
        K,
        maskT,
      );
    const logits = this.call("W_out", this.linear(v, "W_out"));
    const data = await o.read(logits);
    const log_probs = new Float32Array(L * 21);
    for (let i = 0; i < L; i++)
      log_probs.set(logSoftmax(data.subarray(i * 21, i * 21 + 21)), i * 21);
    return {
      logits: data,
      log_probs,
      decoding_order: Int32Array.from(order.flat),
      trace: await this.readTrace(),
    };
  }
  async sample(
    input,
    {
      randn,
      uniforms,
      seed = 42,
      temperature = 0.1,
      trace = false,
      signal,
    } = {},
  ) {
    if (!(temperature > 0 && Number.isFinite(temperature)))
      throw Error("temperature must be positive");
    this.start(trace);
    const rng = seededRandom(seed),
      o = this.o,
      { L, K, E_idx, mask, chain_mask, bias, S: truth } = input;
    const order = this.order(input, randn ?? normalDraws(L, rng));
    const { v, e } = this.encode(input),
      stack = [v, ...Array.from({ length: 3 }, () => o.zeros([L, 128]))],
      hs = o.zeros([L, 128]);
    const S = new Int32Array(L),
      probs = new Float32Array(L * 20),
      logs = new Float32Array(L * 21),
      logitsAll = new Float32Array(L * 21);
    let draw = 0;
    for (const group of order.groups) {
      signal?.throwIfAborted();
      const total = new Float32Array(21);
      for (const t of group.ids) {
        const idx = E_idx.slice(t * K, (t + 1) * K),
          et = o.gather(
            e,
            Int32Array.from({ length: K }, (_, k) => t * K + k),
          );
        const es = o.concat(et, o.gather(hs, idx)),
          bw = o.tensor(
            Float32Array.from(
              idx,
              (j) => mask[t] * (order.rank[j] < order.rank[t]),
            ),
            [K, 1],
          ),
          fw = o.tensor(
            Float32Array.from(
              idx,
              (j) => mask[t] * (order.rank[j] >= order.rank[t]),
            ),
            [K, 1],
          );
        const enc = o.mask(
            o.concat(et, o.zeros([K, 128]), o.gather(v, idx)),
            fw,
          ),
          maskT = o.tensor([mask[t]], [1, 1]);
        for (let l = 0; l < 3; l++) {
          const ev = o.add(
              o.mask(o.concat(es, o.gather(stack[l], idx)), bw),
              enc,
            ),
            vt = o.gather(stack[l], [t]);
          const out = this.dec(vt, ev, "decoder_layers." + l, K, maskT);
          o.scatter(stack[l + 1], t, out);
        }
        const logit = this.call(
            "W_out",
            this.linear(o.gather(stack[3], [t]), "W_out"),
          ),
          data = await o.read(logit),
          lp = logSoftmax(data);
        logitsAll.set(data, t * 21);
        const w = group.weights[group.ids.indexOf(t)];
        for (let a = 0; a < 21; a++) {
          total[a] += w * data[a];
          logs[t * 21 + a] = chain_mask[t] * mask[t] * lp[a];
        }
      }
      const last = group.ids.at(-1),
        p = softmax(
          total,
          temperature,
          bias.subarray(last * 21, last * 21 + 21),
          20,
        );
      let u = uniforms?.[draw++] ?? rng(),
        aa = 19;
      for (let a = 0; a < 20; a++) {
        u -= p[a];
        if (u < 0) {
          aa = a;
          break;
        }
      }
      for (const t of group.ids) {
        const cm = chain_mask[t] * mask[t];
        aa = cm ? aa : truth[t];
        S[t] = aa;
        for (let a = 0; a < 20; a++) probs[t * 20 + a] = cm * p[a];
        o.scatter(hs, t, o.gather(this.w["W_s.weight"], [aa]));
      }
    }
    const score = (m) => {
      let sum = 0,
        n = 0;
      for (let i = 0; i < L; i++) {
        const w = mask[i] * chain_mask[i] * m[i];
        sum -= logs[i * 21 + S[i]] * w;
        n += w;
      }
      return sum / (n + 1e-8);
    };
    return {
      S,
      sequence: Array.from(S, (a) => ALPHABET[a]).join(""),
      sampling_probs: probs,
      log_probs: logs,
      logits: logitsAll,
      decoding_order: Int32Array.from(order.flat),
      global_score: score(mask),
      ligand_score: score(input.mask_XY),
      trace: await this.readTrace(),
    };
  }
  async readTrace() {
    const result = {};
    for (const [k, t] of Object.entries(this.trace))
      result[k] = await this.o.read(t);
    return result;
  }
}
export function makeEngine(ops, bundle, defaults = {}) {
  const model = new LigandModel(ops, bundle);
  let destroyed = false,
    busy = false;
  return {
    model,
    async design(structureText, opts = {}) {
      if (destroyed) throw Error("Engine destroyed");
      if (busy) throw Error("Engine is already running");
      busy = true;
      const begin = performance.now();
      try {
        const options = { ...defaults, ...opts };
        options.signal?.throwIfAborted();
        const parsed = parseStructure(structureText, options),
          input = featurize(parsed, options),
          count = options.numSequences ?? 1;
        if (!Number.isInteger(count) || count < 1)
          throw Error("numSequences must be a positive integer");
        const results = [];
        for (let i = 0; i < count; i++) {
          results.push(
            await model.sample(input, {
              ...options,
              seed: (options.seed ?? 42) + i,
            }),
          );
          ops.clear();
        }
        return {
          sequences: results.map((r) => r.sequence),
          scores: results.map((r) => ({
            global: r.global_score,
            ligand: r.ligand_score,
          })),
          logProbs: results.map((r) => r.log_probs),
          samplingProbs: results.map((r) => r.sampling_probs),
          residues: parsed.residues,
          timingMs: performance.now() - begin,
        };
      } finally {
        busy = false;
        ops.clear();
      }
    },
    destroy() {
      destroyed = true;
      ops.destroy();
    },
  };
}
