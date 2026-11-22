/** Architecture shared by the FP32 reference and GPU operator backends. */
const mod = (x, n) => ((x % n) + n) % n;
export function createRng(seed = 42) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function sample(logits, temp, uniform) {
  if (temp === 0 || temp == null) {
    let k = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[k]) k = i;
    return k;
  }
  const max = Math.max(...logits),
    p = Array.from(logits, (v) => Math.exp((v - max) / temp));
  let sum = p.reduce((a, b) => a + b, 0),
    c = 0;
  for (let i = 0; i < p.length; i++) {
    c += p[i] / sum;
    if (uniform <= c) return i;
  }
  return p.length - 1;
}
export function chiRbf(angle, bins = 72) {
  const a = Float32Array.from({ length: bins }, (_, i) =>
      Math.exp(
        -0.5 *
          (Math.min(
            mod(-180 + i * 5 - angle, 360),
            mod(angle + 180 - i * 5, 360),
          ) /
            2.5) **
            2,
      ),
    ),
    s = a.reduce((a, b) => a + b, 0);
  return a.map((x) => x / s);
}
export class LaserModel {
  constructor(backend, config) {
    this.b = backend;
    this.config = config;
    this.dim = config.model.node_embedding_dim;
    this.v = config.model.num_laser_vectors;
  }
  mlp(x, p) {
    const b = this.b;
    return b.linear(
      b.act(
        b.linear(b.act(b.linear(x, p + ".layers.0"), "gelu"), p + ".layers.3"),
        "gelu",
      ),
      p + ".layers.6",
    );
  }
  gvp(x, p) {
    const b = this.b;
    let h = b.vecLinear(x.v, p + ".wh"),
      s = x.s;
    if (b.weights[p + ".ws.weight"])
      s = b.linear(b.cat([s, b.vecLength(h)]), p + ".ws");
    let v = b.zeros(x.s.r, 0);
    if (b.weights[p + ".wv.weight"]) {
      v = b.vecLinear(h, p + ".wv");
      let gate;
      if (b.weights[p + ".wsv.weight"])
        gate = b.act(b.linear(b.act(s, "sigmoid"), p + ".wsv"), "sigmoid");
      else gate = b.act(b.vecLength(v), "sigmoid");
      v = b.vecScale(v, gate);
    }
    if (b.weights[p + ".ws.weight"]) s = b.act(s, "gelu");
    return { s, v };
  }
  eqnorm(x, p) {
    return {
      s: this.b.weights[p + ".scalar_norm.weight"]
        ? this.b.norm(x.s, p + ".scalar_norm")
        : x.s,
      v: this.b.vecNorm(x.v),
    };
  }
  denseGvp(x, p) {
    return this.gvp(
      this.eqnorm(
        this.gvp(this.eqnorm(this.gvp(x, p + ".l1"), p + ".norm1"), p + ".l2"),
        p + ".norm2",
      ),
      p + ".l3",
    );
  }
  message(src, sink, edge, p) {
    const b = this.b,
      x = b.cat([src, edge, sink]),
      raw = b.linear(x, p + ".gatW"),
      logits = b.atten(raw, p + ".gatA");
    return {
      logits,
      messages: b.weights[p + ".dense_node_update_layers.layers.0.weight"]
        ? this.mlp(x, p + ".dense_node_update_layers")
        : raw,
    };
  }
  edgeUpdate(src, sink, edge, p) {
    const b = this.b;
    if (!b.weights[p + ".linear_edge_updates.layers.0.weight"]) return edge;
    return b.norm(
      b.add(
        edge,
        this.mlp(b.cat([src, edge, sink]), p + ".linear_edge_updates"),
      ),
      p + ".edge_norm",
    );
  }
  homo(nodes, edge, idx, p) {
    const b = this.b,
      m = this.message(
        b.gather(nodes.s, idx[0]),
        b.gather(nodes.s, idx[1]),
        edge,
        p,
      ),
      heads = b.weights[p + ".gatA"].shape[0];
    const s = b.pool(m.messages, m.logits, idx[1], nodes.s.r, heads, this.dim);
    const out = this.eqnorm(
      this.gvp({ s, v: nodes.v }, p + ".vectors_update_layer"),
      p + ".equivariant_layer_norm",
    );
    return [
      out,
      this.edgeUpdate(
        b.gather(out.s, idx[0]),
        b.gather(out.s, idx[1]),
        edge,
        p,
      ),
    ];
  }
  hetero(prev, sources, sinks, edges, indices, p) {
    const b = this.b,
      ms = sources.map((src, i) =>
        this.message(src, sinks[i], edges[i], p + `.subgats.${i}`),
      );
    let s = b.pool(
      b.rows(ms.map((x) => x.messages)),
      b.rows(ms.map((x) => x.logits)),
      indices.flatMap((x) => x),
      prev.s.r,
      1,
      this.dim,
    );
    const q = p + ".dense_residual_node_update";
    s = b.norm(b.add(prev.s, s), q + ".node_norm1");
    s = b.norm(
      b.add(
        s,
        b.linear(b.act(b.linear(s, q + ".layers.0"), "gelu"), q + ".layers.2"),
      ),
      q + ".node_norm2",
    );
    return this.eqnorm(
      this.gvp({ s, v: prev.v }, p + ".vectors_update_layer"),
      p + ".equivariant_layer_norm",
    );
  }
  async encode(batch, { trace = false } = {}) {
    const b = this.b,
      n = batch.length,
      m = batch.ligandCoords.length / 3,
      pp = batch.pp,
      lp = batch.lp,
      ll = batch.ll,
      tr = {};
    const record = (key, value) => {
      if (trace) {
        if (value?.s) {
          tr[key + ".scalars"] = value.s;
          tr[key + ".vectors"] = value.v;
        } else tr[key] = value;
      }
    };
    const ligVec = new Float32Array(m * 3),
      count = new Int32Array(m);
    for (let e = 0; e < ll[0].length; e++) {
      const src = ll[0][e],
        sink = ll[1][e];
      count[sink]++;
      for (let d = 0; d < 3; d++)
        ligVec[sink * 3 + d] += batch.ligandCoords[src * 3 + d];
    }
    for (let i = 0; i < m; i++) {
      let norm = 0;
      for (let d = 0; d < 3; d++) {
        const j = i * 3 + d;
        ligVec[j] = ligVec[j] / Math.max(1, count[i]) - batch.ligandCoords[j];
        norm += ligVec[j] ** 2;
      }
      norm = Math.sqrt(Math.max(norm, 1e-8));
      for (let d = 0; d < 3; d++) ligVec[i * 3 + d] /= norm;
    }
    let lig = this.gvp(
      {
        s: b.linear(
          b.from(batch.ligandNodes, m, 26),
          "ligand_encoder.ligand_input_layer",
        ),
        v: b.from(ligVec, m, 3),
      },
      "ligand_encoder.input_gvp",
    );
    let le = b.linear(
      b.rbf(
        b.from(batch.llDistances, ll[0].length, 1),
        "ligand_encoder.ligand_edge_rbf_encoding.D_mu",
        0.2,
      ),
      "ligand_encoder.ligand_edge_input_layer",
    );
    for (let i = 0; i < 3; i++) {
      [lig, le] = this.homo(lig, le, ll, `ligand_encoder.gat_layers.${i}`);
      record(`ligand_encoder.gat_layers.${i}.0`, lig);
      record(`ligand_encoder.gat_layers.${i}.1`, le);
    }
    record("ligand_encoder", lig);
    lig = this.denseGvp(lig, "ligand_encoder_output_gvp");
    record("ligand_encoder_output_gvp", lig);
    const frame = new Float32Array(n * 12),
      bb = batch.backbone;
    const norm = (v) => {
      const s = Math.hypot(...v);
      return v.map((x) => x / s);
    };
    for (let i = 0; i < n; i++) {
      const at = (k) =>
          Array.from(bb.slice(i * 15 + k * 3, i * 15 + k * 3 + 3)),
        ca = at(1),
        sub = (k) => at(k).map((x, d) => x - ca[d]);
      const nc = norm(sub(0)),
        cc = norm(sub(3)),
        bis = norm(nc.map((x, d) => -x - cc[d])),
        cb = norm(sub(2));
      frame.set([...bis, ...cb, ...nc, ...cc], i * 12);
    }
    let prot = this.eqnorm(
      this.gvp(
        { s: b.zeros(n, this.dim), v: b.from(frame, n, 12) },
        "backbone_frame_vec_input_layer",
      ),
      "backbone_frame_vec_norm",
    );
    record("backbone_frame_vec_norm", prot);
    const disp = b.from(
      Float32Array.from(
        lp[0].flatMap((l, e) =>
          Array.from(
            { length: 3 },
            (_, d) => batch.ligandCoords[3 * l + d] - bb[lp[1][e] * 15 + 3 + d],
          ),
        ),
      ),
      lp[0].length,
      3,
    );
    const pdots = () =>
      b.dots(b.gather(prot.v, pp[0]), b.gather(prot.v, pp[1]));
    const ldots = () => b.dots(b.gather(prot.v, lp[1]), disp);
    let pe = b.linear(
      b.cat([
        b.rbf(
          b.from(batch.ppDistances, pp[0].length, 25),
          "prot_prot_rbf_encoding.D_mu",
          1.25,
        ),
        pdots(),
      ]),
      "prot_prot_edge_input_layer",
    );
    let ple = b.linear(
      b.cat([
        b.rbf(
          b.from(batch.lpDistances, lp[0].length, 5),
          "lig_prot_rbf_encoding.D_mu",
          0.2,
        ),
        ldots(),
      ]),
      "lig_prot_edge_input_layer",
    );
    record("prot_prot_edge_input_layer", pe);
    record("lig_prot_edge_input_layer", ple);
    for (let i = 0; i < 3; i++) {
      const p = `protein_encoder_layers.${i}`,
        old = prot;
      prot = this.hetero(
        old,
        [b.gather(old.s, pp[0]), b.gather(lig.s, lp[0])],
        [b.gather(old.s, pp[1]), b.gather(old.s, lp[1])],
        [pe, ple],
        [pp[1], lp[1]],
        p + ".hetgat",
      );
      // HeteroGAT's subgats have update_edges=false: encoder edges are updated only by these final angular projections.
      pe = b.linear(b.cat([pe, pdots()]), p + ".final_pr_pr_mlp");
      ple = b.linear(b.cat([ple, ldots()]), p + ".final_lig_pr_mlp");
      record(p + ".0", prot);
      record(p + ".1", pe);
      record(p + ".2", ple);
    }
    const traces = {};
    if (trace)
      for (const [k, v] of Object.entries(tr)) traces[k] = await b.read(v);
    return { lig, prot, pe, ple, traces };
  }
  async run(batch, opts = {}) {
    const b = this.b,
      n = batch.length,
      trace = opts.trace ?? false,
      rng = opts.rng ?? batch.rng ?? createRng(opts.seed ?? 42),
      mark = b.mark();
    try {
      const enc = await this.encode(batch, { trace }),
        { lig, prot, pe, ple } = enc;
      const order =
        opts.decodingOrder ??
        batch.decodingOrder ??
        Array.from({ length: n }, (_, i) => i).sort(
          (a, c) => (batch.fixed?.[c] ?? 0) - (batch.fixed?.[a] ?? 0),
        );
      const stack = [
        prot,
        ...Array.from({ length: 3 }, () => ({
          s: b.zeros(n, this.dim),
          v: b.zeros(n, this.v * 3),
        })),
      ];
      const labels = new Int32Array(n).fill(21),
        chis = new Float32Array(n * 4).fill(NaN),
        chiEnc = Array.from({ length: n }, () => new Float32Array(288)),
        decoded = new Set();
      const chiBins = new Int32Array(n * 4).fill(-1),
        logits = new Float32Array(n * 21),
        chiLogits = new Float32Array(n * 288),
        rawSequence = [],
        rawChi = Array.from({ length: 4 }, () => []);
      let drawIndex = 0,
        choiceIndex = 0;
      const draw = () => (opts.uniforms ? opts.uniforms[drawIndex++] : rng());
      const gpuPipelined =
        typeof b.readMany === "function" &&
        typeof b.sampleIndex === "function" &&
        typeof b.catOneHot === "function" &&
        typeof b.chiAngle === "function" &&
        typeof b.chiRbf === "function" &&
        typeof b.maskRow === "function";
      if (gpuPipelined) {
        const omitSet = new Set("X" + (opts.omitAAs ?? ""));
        const omitMaskData = new Float32Array(21).fill(1);
        for (const aa of omitSet) {
          const k = this.config.alphabet.indexOf(aa);
          if (k < 0) throw Error(`Unknown amino acid ${aa}`);
          omitMaskData[k] = 0;
        }
        const omitMaskGpu = b.from(omitMaskData, 1, 21);
        const seqEmbedWeight = b.weight("sequence_label_embedding.weight");
        const chiMaskTable = b.weights["rotamer_builder.aa_to_chi_angle_mask"].data;
        const chiMaskGpu = b.from(
          Float32Array.from({ length: 22 * 4 }, (_, idx) => {
            const aa = Math.floor(idx / 4);
            const k = idx % 4;
            return chiMaskTable[(aa >= 20 ? 7 : aa) * 4 + k] ? 1 : 0;
          }),
          22,
          4,
        );
        const labelsGpu = b.from(new Float32Array(n + 1).fill(21), n + 1, 1);
        const chiEncFlat = b.zeros((n + 1) * 4, 72);
        const chiEncView = { buffer: chiEncFlat.buffer, r: n + 1, c: 288 };
        const allSeqLogitsGpu = b.zeros(n, 21);
        const allChiLogitsGpu = b.zeros(n * 4, 72);
        const allChiBinsGpu = b.zeros(n * 4, 1);
        const allChiAnglesGpu = b.zeros(n * 4, 1);

        for (const node of order) {
          opts.signal?.throwIfAborted();
          const stepMark = b.mark();
          const pEdges = batch.pp[1]
              .map((s, e) => (s === node ? e : -1))
              .filter((e) => e >= 0),
            lEdges = batch.lp[1]
              .map((s, e) => (s === node ? e : -1))
              .filter((e) => e >= 0);
          const masked = pEdges.filter((e) => !decoded.has(batch.pp[0][e])),
            unmasked = pEdges.filter((e) => decoded.has(batch.pp[0][e])),
            edges = [...masked, ...unmasked],
            src = edges.map((e) => batch.pp[0][e]);
          const gatherIdx = src.map((s, i) => (i < masked.length ? n : s));
          const seq = b.gather(seqEmbedWeight, b.gather(labelsGpu, gatherIdx));
          const prevChi = b.gather(chiEncView, gatherIdx);
          const prefix = b.cat([seq, prevChi, b.gather(pe, edges)]),
            ligSource = b.gather(
              lig.s,
              lEdges.map((e) => batch.lp[0][e]),
            ),
            ligEdge = b.gather(ple, lEdges);
          for (let i = 0; i < 3; i++) {
            const allSrc = b.rows([
              b.gather(
                prot.s,
                masked.map((e) => batch.pp[0][e]),
              ),
              b.gather(
                stack[i].s,
                unmasked.map((e) => batch.pp[0][e]),
              ),
            ]);
            const prev = {
              s: b.gather(stack[i].s, [node]),
              v: b.gather(stack[i].v, [node]),
            };
            const out = this.hetero(
              prev,
              [b.cat([prefix, allSrc]), ligSource],
              [
                b.gather(
                  prev.s,
                  edges.map(() => 0),
                ),
                b.gather(
                  prev.s,
                  lEdges.map(() => 0),
                ),
              ],
              [b.zeros(edges.length, 0), ligEdge],
              [edges.map(() => 0), lEdges.map(() => 0)],
              `protein_decoder_layers.${i}.hetgat`,
            );
            b.assign(stack[i + 1].s, node, out.s);
            b.assign(stack[i + 1].v, node, out.v);
          }
          let curr = {
            s: b.gather(stack[3].s, [node]),
            v: b.gather(stack[3].v, [node]),
          };
          const seqLogitsGpu = b.linear(curr.s, "sequence_output_layer");
          b.assign(allSeqLogitsGpu, node, seqLogitsGpu);
          const fixed = batch.fixed?.[node];
          const seqDraw = draw();
          const seqOverride = opts.choices ? opts.choices[choiceIndex++] : null;
          const labelGpu = fixed
            ? b.from([batch.sequence[node]], 1, 1)
            : seqOverride !== null
              ? b.from([seqOverride], 1, 1)
              : b.sampleIndex(
                  seqLogitsGpu,
                  opts.temperature ?? 0.1,
                  seqDraw,
                  omitMaskGpu,
                );
          b.assign(labelsGpu, node, labelGpu);
          const embed = b.gather(seqEmbedWeight, labelGpu);
          const labelMaskGpu = b.gather(chiMaskGpu, labelGpu);
          let prevGpu = b.zeros(1, 0);
          for (let i = 0; i < 4; i++) {
            const chiLogitsGpu = this.mlp(
              i === 0 ? b.cat([curr.s, embed]) : b.cat([curr.s, embed, prevGpu]),
              `chi_prediction_layers.${i}`,
            );
            b.assign(allChiLogitsGpu, node * 4 + i, chiLogitsGpu);
            const chiDraw = draw();
            const choiceGpu = opts.choices
              ? b.from([opts.choices[choiceIndex++]], 1, 1)
              : b.sampleIndex(chiLogitsGpu, opts.chiTemperature ?? 0.1, chiDraw);
            b.assign(allChiBinsGpu, node * 4 + i, choiceGpu);
            let angleGpu;
            if (fixed && Number.isFinite(batch.chi?.[node * 4 + i])) {
              angleGpu = b.from([batch.chi[node * 4 + i]], 1, 1);
            } else {
              const offsetGpu = this.mlp(
                b.catOneHot(chiLogitsGpu, choiceGpu),
                `chi_offset_prediction_layers.${i}`,
              );
              angleGpu = b.chiAngle(choiceGpu, offsetGpu);
            }
            b.assign(allChiAnglesGpu, node * 4 + i, angleGpu);
            const encodingGpu = b.chiRbf(angleGpu);
            b.assign(
              chiEncFlat,
              node * 4 + i,
              b.maskRow(encodingGpu, labelMaskGpu, i),
            );
            prevGpu = i === 0 ? encodingGpu : b.cat([prevGpu, encodingGpu]);
            if (i < 3) {
              curr = this.eqnorm(
                this.gvp(
                  { s: b.cat([curr.s, prevGpu]), v: curr.v },
                  `chi_vector_update_layers.${i}`,
                ),
                `chi_vector_layer_norms.${i}`,
              );
            }
          }
          decoded.add(node);
          b.releaseSince(stepMark);
          if (
            decoded.size % 16 === 0 &&
            decoded.size < n &&
            typeof b.sync === "function"
          ) {
            await b.sync();
          }
          opts.onProgress?.({
            phase: "design",
            completed: decoded.size,
            total: n,
          });
        }

        const [
          labelsData,
          allSeqLogitsData,
          allChiLogitsData,
          allChiBinsData,
          allChiAnglesData,
        ] = await b.readMany([
          labelsGpu,
          allSeqLogitsGpu,
          allChiLogitsGpu,
          allChiBinsGpu,
          allChiAnglesGpu,
        ]);

        for (let node = 0; node < n; node++) {
          labels[node] = Math.round(labelsData[node]);
          const seqSlice = allSeqLogitsData.subarray(node * 21, (node + 1) * 21);
          logits.set(seqSlice, node * 21);
          if (!batch.fixed?.[node]) {
            for (const aa of omitSet) {
              const k = this.config.alphabet.indexOf(aa);
              logits[node * 21 + k] = -3.4028234663852886e38;
            }
          }
          for (let i = 0; i < 4; i++) {
            const mask =
              chiMaskTable[(labels[node] === 20 ? 7 : labels[node]) * 4 + i];
            if (mask) {
              chiBins[node * 4 + i] = Math.round(allChiBinsData[node * 4 + i]);
              chis[node * 4 + i] = allChiAnglesData[node * 4 + i];
              chiLogits.set(
                allChiLogitsData.subarray(
                  (node * 4 + i) * 72,
                  (node * 4 + i + 1) * 72,
                ),
                node * 288 + i * 72,
              );
            }
          }
        }
        if (trace) {
          for (const node of order) {
            rawSequence.push(
              ...allSeqLogitsData.subarray(node * 21, (node + 1) * 21),
            );
            for (let i = 0; i < 4; i++) {
              rawChi[i].push(
                ...allChiLogitsData.subarray(
                  (node * 4 + i) * 72,
                  (node * 4 + i + 1) * 72,
                ),
              );
            }
          }
        }
      } else {
        for (const node of order) {
        opts.signal?.throwIfAborted();
        const stepMark = b.mark();
        const pEdges = batch.pp[1]
            .map((s, e) => (s === node ? e : -1))
            .filter((e) => e >= 0),
          lEdges = batch.lp[1]
            .map((s, e) => (s === node ? e : -1))
            .filter((e) => e >= 0);
        const masked = pEdges.filter((e) => !decoded.has(batch.pp[0][e])),
          unmasked = pEdges.filter((e) => decoded.has(batch.pp[0][e])),
          edges = [...masked, ...unmasked],
          src = edges.map((e) => batch.pp[0][e]);
        const seq = b.gather(
          b.weight("sequence_label_embedding.weight"),
          src.map((s, i) => (i < masked.length ? 21 : labels[s])),
        );
        const prevChi = b.from(
          Float32Array.from(
            src.flatMap((s, i) =>
              Array.from(i < masked.length ? new Float32Array(288) : chiEnc[s]),
            ),
          ),
          src.length,
          288,
        );
        const prefix = b.cat([seq, prevChi, b.gather(pe, edges)]),
          ligSource = b.gather(
            lig.s,
            lEdges.map((e) => batch.lp[0][e]),
          ),
          ligEdge = b.gather(ple, lEdges);
        for (let i = 0; i < 3; i++) {
          const allSrc = b.rows([
            b.gather(
              prot.s,
              masked.map((e) => batch.pp[0][e]),
            ),
            b.gather(
              stack[i].s,
              unmasked.map((e) => batch.pp[0][e]),
            ),
          ]);
          const prev = {
            s: b.gather(stack[i].s, [node]),
            v: b.gather(stack[i].v, [node]),
          };
          const out = this.hetero(
            prev,
            [b.cat([prefix, allSrc]), ligSource],
            [
              b.gather(
                prev.s,
                edges.map(() => 0),
              ),
              b.gather(
                prev.s,
                lEdges.map(() => 0),
              ),
            ],
            [b.zeros(edges.length, 0), ligEdge],
            [edges.map(() => 0), lEdges.map(() => 0)],
            `protein_decoder_layers.${i}.hetgat`,
          );
          b.assign(stack[i + 1].s, node, out.s);
          b.assign(stack[i + 1].v, node, out.v);
        }
        let curr = {
          s: b.gather(stack[3].s, [node]),
          v: b.gather(stack[3].v, [node]),
        };
        const raw = await b.read(b.linear(curr.s, "sequence_output_layer"));
        rawSequence.push(...raw);
        const fixed = batch.fixed?.[node];
        if (!fixed)
          for (const aa of new Set("X" + (opts.omitAAs ?? ""))) {
            const k = this.config.alphabet.indexOf(aa);
            if (k < 0) throw Error(`Unknown amino acid ${aa}`);
            raw[k] = -3.4028234663852886e38;
          }
        const seqDraw = draw(),
          sampled = opts.choices
            ? opts.choices[choiceIndex++]
            : sample(raw, opts.temperature ?? 0.1, seqDraw);
        labels[node] = fixed ? batch.sequence[node] : sampled;
        logits.set(raw, node * 21);
        const embed = b.gather(b.weight("sequence_label_embedding.weight"), [
            labels[node],
          ]),
          prev = [];
        for (let i = 0; i < 4; i++) {
          const log = b.read(
            this.mlp(
              b.cat([
                curr.s,
                embed,
                b.from(Float32Array.from(prev), 1, prev.length),
              ]),
              `chi_prediction_layers.${i}`,
            ),
          );
          const values = await log;
          rawChi[i].push(...values);
          const chiDraw = draw(),
            choice = opts.choices
              ? opts.choices[choiceIndex++]
              : sample(values, opts.chiTemperature ?? 0.1, chiDraw),
            onehot = new Float32Array(72);
          onehot[choice] = 1;
          const offset = (
            await b.read(
              this.mlp(
                b.from(Float32Array.from([...values, ...onehot]), 1, 144),
                `chi_offset_prediction_layers.${i}`,
              ),
            )
          )[0];
          let angle = Math.fround(
            mod(Math.fround(-180 + choice * 5 + offset) + 180, 360) - 180,
          );
          if (fixed && Number.isFinite(batch.chi?.[node * 4 + i]))
            angle = batch.chi[node * 4 + i];
          const encoding = chiRbf(angle);
          prev.push(...encoding);
          const mask =
            b.weights["rotamer_builder.aa_to_chi_angle_mask"].data[
              (labels[node] === 20 ? 7 : labels[node]) * 4 + i
            ];
          if (mask) {
            chiBins[node * 4 + i] = choice;
            chis[node * 4 + i] = angle;
            chiEnc[node].set(encoding, i * 72);
            chiLogits.set(values, node * 288 + i * 72);
          }
          if (i < 3)
            curr = this.eqnorm(
              this.gvp(
                {
                  s: b.cat([
                    curr.s,
                    b.from(Float32Array.from(prev), 1, prev.length),
                  ]),
                  v: curr.v,
                },
                `chi_vector_update_layers.${i}`,
              ),
              `chi_vector_layer_norms.${i}`,
            );
        }
        decoded.add(node);
        b.releaseSince(stepMark);
        opts.onProgress?.({
          phase: "design",
          completed: decoded.size,
          total: n,
        });
        }
      }
      const result = {
        sequence: labels,
        chiAngles: chis,
        chiBins,
        sequenceLogits: logits,
        chiLogits,
        traces: enc.traces,
      };
      if (trace) {
        result.traces.sequence_output_layer = Float32Array.from(rawSequence);
        rawChi.forEach(
          (x, i) =>
            (result.traces[`chi_prediction_layers.${i}`] =
              Float32Array.from(x)),
        );
      }
      return result;
    } finally {
      b.releaseSince(mark);
    }
  }
}
