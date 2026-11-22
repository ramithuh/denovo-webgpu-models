import { createEsmcEngine, requestWebGpuDevice } from "../webgpu/index.js";
import { scoresSource, softmaxSource, streamingContextSource } from "../webgpu/shaders.js";
import { diffStats } from "../shared/weights.js";
import { check, oracle } from "./parity.js";

export async function run() {
  const { device, info } = await requestWebGpuDevice({ useF16: false });
  const engine = await createEsmcEngine({
    device, baseUrl: "/weights/esmc-300m", precision: "fp16", cache: "none",
    useF16: false,
  });
  try {
    const results = [];
    for (const length of [63, 64, 71]) {
      const tokens = Int32Array.from({ length }, (_, i) => 4 + i % 20);
      const ids = Int32Array.from(tokens, (_, i) => i === length - 1 ? 0 : i % 2 ? -1 : 7);
      const baseline = await engine.forward(tokens, { sequenceIds: ids, returnAttentions: true });
      const shape = engine.config.n_heads * length * length;
      check(baseline.attentions.length === engine.config.n_layers &&
        baseline.attentions.every(a => a.length === shape), "Attention trace shape changed");
      // The singleton final token must attend exclusively to itself for every head.
      for (const attention of baseline.attentions) {
        for (let h = 0; h < engine.config.n_heads; h++) {
          const start = (h * length + length - 1) * length;
          check(attention[start] === 0 && attention[start + length - 1] === 1, "ID mask changed");
        }
      }
      const dispatch = engine.dispatch;
      const { d_model: D, n_heads: H } = engine.config;
      const scores = scoresSource(length, D, H), softmax = softmaxSource(length, H);
      const stream = streamingContextSource(length, H);
      let quadratic = 0, streaming = 0;
      engine.dispatch = function (code, inputs, elements, groups) {
        if (code === scores || code === softmax) quadratic++;
        if (code === stream) streaming++;
        return dispatch.call(this, code, inputs, elements, groups);
      };
      let actual;
      try {
        actual = await engine.forward(tokens, { sequenceIds: ids });
      } finally {
        engine.dispatch = dispatch;
      }
      check(actual.attentions === undefined, "Unexpected attention trace");
      check(quadratic === (length < 64 ? 2 * engine.config.n_layers : 0), "Unexpected quadratic scratch");
      check(streaming === (length < 64 ? 0 : engine.config.n_layers), "Unexpected streaming dispatch");
      const metrics = {};
      for (const name of ["logits", "embeddings", "pooled", "llr", "canonicalProbabilities"]) {
        const stats = diffStats(actual[name], baseline[name]);
        check(stats.relRms < 1e-4, name + ": " + JSON.stringify(stats));
        metrics[name] = stats.relRms;
      }
      results.push({ length, quadratic, metrics });
    }
    const prefix = "/fixtures/oracle_fp16_ubiquitin";
    const meta = await (await fetch(prefix + ".json")).json();
    const reference = oracle(meta, await (await fetch(prefix + ".bin")).arrayBuffer());
    const normal = await engine.forward(meta.sequence);
    const oracleRelRms = diffStats(normal.logits, reference("wt/logits")).relRms;
    check(oracleRelRms < 1e-4, "Oracle logits exceeded tolerance: " + oracleRelRms);
    return { vendor: info.vendor, architecture: info.architecture, results, oracleRelRms };
  } finally {
    engine.destroy();
    device.destroy();
  }
}
