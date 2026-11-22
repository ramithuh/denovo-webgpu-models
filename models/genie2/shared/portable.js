// A small source bridge for handwritten scalar algorithms, never a traced graph.
// WGSL remains the source of truth; the CPU reference uses the same control flow.
export function scalarJS(source) {
  return source
    .replace(
      /var\s+(\w+)\s*:\s*array<f32,\s*(\d+)>;/g,
      "let $1=new Float32Array($2);",
    )
    .replace(
      /var\s+(\w+)\s*:\s*array<i32,\s*(\d+)>;/g,
      "let $1=new Int32Array($2);",
    )
    .replace(/\b(let|var)\s+(\w+)\s*:\s*(?:f32|u32|i32)/g, "let $2")
    .replace(/\bvar\b/g, "let")
    .replace(/(\d+)[ui]\b/g, "$1")
    .replace(/\bloop\s*\{/g, "while(true){")
    .replace(/\bf32\(/g, "Number(")
    .replace(/\bi32\(/g, "Math.trunc(")
    .replace(/\bu32\(/g, "Math.trunc(")
    .replace(
      /\b(abs|sqrt|sin|cos|pow|exp|log|max|min|floor|ceil)\(/g,
      "Math.$1(",
    )
    .replace(/\bfma\(/g, "rfFma(")
    .replace(/\brfdiv\(/g, "rfdiv(");
}
export function portable(o, shape, inputs, source) {
  return o.kernel(shape, inputs, scalarJS(source), source);
}
