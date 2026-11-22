/** Gather contiguous rows of a row-major float32 embedding table. */
export function gatherEmbeddings(ids, weights, width) {
  if (!(weights instanceof Float32Array) || !Number.isSafeInteger(width) || width <= 0 ||
      weights.length % width !== 0) {
    throw new TypeError('Expected a float32 table with complete rows');
  }
  const rows = weights.length / width;
  const result = new Float32Array(ids.length * width);
  for (let position = 0; position < ids.length; position++) {
    const id = ids[position];
    if (!Number.isInteger(id) || id < 0 || id >= rows) {
      throw new RangeError(`Invalid embedding row at token position ${position}`);
    }
    result.set(weights.subarray(id * width, (id + 1) * width), position * width);
  }
  return result;
}
