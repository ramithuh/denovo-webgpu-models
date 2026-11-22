import { sha256 } from "../shared/weights.js";
export function assert(value, message = "Assertion failed") {
  if (!value) throw new Error(message);
}
export function stats(a, b) {
  assert(a.length === b.length, "Shape mismatch");
  let sum = 0,
    ref = 0,
    maxAbs = 0;
  for (let i = 0; i < a.length; i++) {
    assert(Number.isFinite(a[i]) && Number.isFinite(b[i]), `Nonfinite at ${i}`);
    sum += (a[i] - b[i]) ** 2;
    ref += b[i] ** 2;
    maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i]));
  }
  return { relRMS: Math.sqrt(sum / Math.max(ref, 1e-30)), maxAbs };
}
export async function oracle(precision, read) {
  const meta = JSON.parse(
      new TextDecoder().decode(await read(`fixtures/oracle_${precision}.json`)),
    ),
    blob = await read(meta.binary.file);
  assert(
    blob.byteLength === meta.binary.byteLength &&
      (await sha256(blob)) === meta.binary.sha256,
    "Oracle checksum mismatch",
  );
  return {
    meta,
    tensor(name) {
      const r = meta.binary.tensors[name];
      assert(r, `Missing fixture ${name}`);
      return new Float32Array(blob, r.offset, r.byteLength / 4);
    },
  };
}
export function parity(actual, expected, label) {
  const result = stats(actual, expected);
  assert(
    result.relRMS < 1e-4,
    `${label}: relRMS=${result.relRMS}, maxAbs=${result.maxAbs}`,
  );
  return result;
}
export async function checkEngine(engine, ref, log = () => {}) {
  const result = { A: {}, B: {}, C: {} };
  for (const c of ref.meta.A) {
    const out = await engine.forward(c.sequence, {
      returnIntermediates: true,
      inputMask: c.inputMask,
    });
    const trace = {};
    for (const [name, key] of Object.entries(c.trace))
      trace[name] = parity(
        out.intermediates[name],
        ref.tensor(key),
        `A.${c.name}.${name}`,
      );
    parity(out.logits, ref.tensor(c.trace.logits), "final logits");
    result.A[c.name] = { trace, timingMs: out.timingMs };
    log(`A ${c.name}: ${trace.logits.relRMS}`);
  }
  for (const mode of ["B", "C"]) {
    const c = ref.meta[mode],
      input =
        mode === "B"
          ? { length: c.initialSequence.length }
          : {
              fixedSequence: c.initialSequence.replaceAll("#", "X"),
              maskPositions: c.order,
            };
    const out = await engine.generate({
      ...input,
      order: c.order,
      uniforms: c.uniforms,
    });
    assert(
      out.sequence === c.sequence,
      `${mode}: sequence mismatch ${out.sequence} != ${c.sequence}`,
    );
    assert(
      Array.from(out.tokens).toString() === c.tokens.toString(),
      `${mode}: token mismatch`,
    );
    const steps = out.steps.map((s, i) => {
      assert(
        s.token === c.steps[i].token && s.position === c.steps[i].position,
        `${mode}: step token/order`,
      );
      return parity(s.logits, ref.tensor(c.steps[i].logits), `${mode}.${i}`);
    });
    result[mode] = {
      sequence: out.sequence,
      exact: true,
      steps,
      timingMs: out.timingMs,
    };
    log(`${mode}: ${out.sequence}`);
  }
  return result;
}
